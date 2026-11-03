"""data/의 대표 예시로 Pythia를 파인튜닝하고, 끝나면 다음 줄 정확도를 남긴다.

사용: uv run finetune.py --model EleutherAI/pythia-14m --steps 300 --lr 3e-4
결과: ckpt/<모델 이름>-ft/ (baseline.py, eval_lines.py, serve_pc.py에 경로로 넘길 수 있다), 로그 logs/finetune-<시각>.log
"""
import argparse
import logging
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_lines import evaluate, load_examples
from log_setup import setup

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-14m")
    ap.add_argument("--steps", type=int, default=300, help="예시 8개 전체를 한 배치로 몇 번 학습할지")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    log = setup("finetune")
    log.info("설정 %s, torch %s, transformers %s", vars(a), torch.__version__, transformers.__version__)

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    # transformers 5는 체크포인트 dtype(Pythia는 float16)으로 불러온다. float16 그대로 AdamW 학습하면 loss가 nan
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(device)

    # 예시 끝에 EOS를 붙여 "여기서 끝난다"도 배우게 한다. 패딩 자리는 loss에서 뺀다
    texts = load_examples()
    batch = tok([t + tok.eos_token for t in texts], return_tensors="pt", padding=True).to(device)
    labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
    log.info("%s: 예시 %d개, 최대 %d토큰, %s", a.model, len(texts), batch["input_ids"].shape[1], device)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    for step in range(1, a.steps + 1):
        loss = model(**batch, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        # 파일엔 매 스텝(DEBUG), 콘솔엔 20스텝마다(INFO)
        lv = logging.INFO if step == 1 or step % 20 == 0 or step == a.steps else logging.DEBUG
        log.log(lv, "step %4d  loss %.4f", step, loss.item())
        if not torch.isfinite(loss):
            log.error("step %d: loss가 %s, 학습 중단", step, loss.item())
            return

    out = a.out or HERE / "ckpt" / f"{a.model.split('/')[-1]}-ft"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log.info("저장: %s", out)

    model.eval()
    r = evaluate(model, tok, texts, device)
    log.info("다음 줄 정확도 %s (%d/%d)", r["line_exact"], r["hit"], r["lines"])
    for want, got in r["misses"]:
        log.info("  틀림: 기대 %r, 생성 %r", want, got)


if __name__ == "__main__":
    main()
