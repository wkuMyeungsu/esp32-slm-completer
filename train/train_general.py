"""범용 학습: MBPP 학습 문제 + 대표 예시로 원본 Pythia를 학습한다. 학습 전에 어휘를 줄여 결과가 바로 보드용이다.

1. 어휘: 학습 코드에서 자주 나오는 토큰 K개 + 1바이트 토큰 + EOS (bridge/vocab_map.py 규칙).
2. 모델: embed_in·embed_out에서 그 행만 남긴다 (50,304 → 약 1,000).
3. 데이터: 프로그램마다 한 줄 시퀀스, 어휘에 없는 토큰은 쪼갠다 (브리지·평가와 같은 규칙). 끝에 EOS.
   대표 예시는 비중이 너무 작아지지 않게 --repeat번 넣는다.
4. heldout(평가 전용 MBPP 100문제) loss가 가장 낮은 스텝을 저장한다.

사용: uv run train_general.py --k 768 --steps 1500
결과: ckpt/<이름>/ (줄인 모델 + keep.json), 로그 logs/train_general-<시각>.log
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "bridge"))

from corpus import examples, mbpp  # noqa: E402
from log_setup import setup  # noqa: E402
from vocab_map import Vocab, token_bytes  # noqa: E402


def build_keep(tok, texts: list[str], k: int) -> list[int]:
    counts = {}
    for t in texts:
        for i in tok(t)["input_ids"]:
            counts[i] = counts.get(i, 0) + 1
    single = {i for i in range(len(tok)) if len(token_bytes(tok.convert_ids_to_tokens(i))) == 1}
    top = [i for i, _ in sorted(counts.items(), key=lambda x: -x[1]) if i not in single][:k]
    return sorted(set(top) | single | {tok.eos_token_id})


def shrink(model, keep: list[int]) -> None:
    """입력·출력 임베딩에서 keep 행만 남긴다. 줄인 id n = keep[n]."""
    # transformers 5는 출력층 속성 이름이 embed_out이 아니다. 이름 대신 표준 접근 함수를 쓴다
    idx = torch.tensor(keep)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    new_in = torch.nn.Embedding(len(keep), emb.embedding_dim)
    new_in.weight.data = emb.weight.data[idx].clone()
    new_out = torch.nn.Linear(head.in_features, len(keep), bias=False)
    new_out.weight.data = head.weight.data[idx].clone()
    model.set_input_embeddings(new_in)
    model.set_output_embeddings(new_out)
    model.config.vocab_size = len(keep)


def batches(seqs: list[list[int]], size: int, pad: int, rng: random.Random, device: str):
    order = list(range(len(seqs)))
    while True:
        rng.shuffle(order)
        for s in range(0, len(order) - size + 1, size):
            yield collate([seqs[i] for i in order[s:s + size]], pad, device)


def collate(group: list[list[int]], pad: int, device: str):
    n = max(map(len, group))
    ids = torch.full((len(group), n), pad)
    labels = torch.full((len(group), n), -100)
    for j, s in enumerate(group):
        ids[j, :len(s)] = torch.tensor(s)
        labels[j, :len(s)] = torch.tensor(s)
    return ids.to(device), (labels != -100).to(device), labels.to(device)  # 패딩 자리만 가린다


@torch.no_grad()
def eval_loss(model, seqs, pad, device) -> float:
    model.eval()
    tot = n = 0.0
    for s in range(0, len(seqs), 32):
        ids, mask, labels = collate(seqs[s:s + 32], pad, device)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        k = (labels[:, 1:] != -100).sum().item()
        tot += out.loss.item() * k
        n += k
    model.train()
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-14m")
    ap.add_argument("--k", type=int, default=768, help="자주 나오는 토큰 몇 개를 남길지 (1바이트 토큰·EOS는 따로)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--repeat", type=int, default=8, help="대표 예시를 몇 번 넣을지")
    ap.add_argument("--max-len", type=int, default=256, help="이보다 긴 프로그램은 뒤를 자른다")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    log = setup("train_general")
    log.info("설정 %s, torch %s, transformers %s", vars(a), torch.__version__, transformers.__version__)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.base)
    train_texts, held_texts = mbpp("train"), mbpp("heldout")
    keep = build_keep(tok, train_texts + examples(), a.k)
    v = Vocab(keep, tok.convert_ids_to_tokens)
    eos = v.remap[tok.eos_token_id]

    def to_seq(text):
        ids, _ = v.encode_ids(tok(text)["input_ids"])
        return (ids + [eos])[:a.max_len]

    train = [to_seq(t) for t in train_texts] + [to_seq(t) for t in examples()] * a.repeat
    held = [to_seq(t) for t in held_texts]
    log.info("어휘 %d (K %d), 학습 시퀀스 %d (MBPP %d + 예시 %d×%d), %d토큰, heldout %d, %s",
             len(keep), a.k, len(train), len(train_texts), len(examples()), a.repeat, sum(map(len, train)),
             len(held), device)

    # transformers 5는 체크포인트 dtype(Pythia는 float16)으로 불러온다. float16 그대로 AdamW 학습하면 loss가 nan
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    shrink(model, keep)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=max(0.05, 2 / a.steps))  # 짧은 시험 실행에서 워밍업이 0스텝이면 0으로 나눔
    out = HERE / "ckpt" / (a.name or f"{a.base.split('/')[-1]}-general-k{a.k}")
    best = float("inf")
    log.info("heldout loss (학습 전) %.4f", eval_loss(model, held, eos, device))
    it = batches(train, a.batch, eos, rng, device)
    for step in range(1, a.steps + 1):
        ids, mask, labels = next(it)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        if not torch.isfinite(loss):
            log.error("step %d: loss가 %s, 학습 중단", step, loss.item())
            return
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        log.log(logging.INFO if step % 50 == 0 or step == 1 else logging.DEBUG, "step %4d  loss %.4f  lr %.2e",
                step, loss.item(), sched.get_last_lr()[0])
        if step % a.eval_every == 0 or step == a.steps:
            hl = eval_loss(model, held, eos, device)
            mark = ""
            if hl < best:
                best = hl
                model.save_pretrained(out)
                tok.save_pretrained(out)
                (out / "keep.json").write_text(json.dumps({"keep": keep, "step": step, "heldout_loss": hl}),
                                               encoding="utf-8")
                mark = " (저장)"
            log.info("step %4d  heldout loss %.4f%s", step, hl, mark)
    log.info("가장 낮은 heldout loss %.4f, 저장: %s", best, out)


if __name__ == "__main__":
    main()
