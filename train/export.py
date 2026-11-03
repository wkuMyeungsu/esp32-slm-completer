"""학습한 Pythia를 보드용 model.bin으로 내보낸다.

체크포인트 두 종류:
  keep.json이 있다 (train_general.py): 학습 전에 어휘를 줄인 모델. 그대로 쓰고, 어휘에 없는 토큰은 쪼갠다 (fallback).
  keep.json이 없다 (finetune.py): 전체 어휘 모델. 대표 예시에 나온 토큰만 남기고, 어휘에 없는 토큰은 버린다.
1. model.bin을 쓰고 npmodel.py로 다시 읽어 torch 출력과 비교한다 (파일 형식까지 확인).
2. eval_bin.py와 같은 평가 (예시 전체, heldout 앞 20문제).

사용: uv run export.py ckpt/pythia-14m-general-k768
결과: ../firmware/model.bin (보드 model 파티션에 굽는다, 형식은 npmodel.py),
      ../bridge/vocab.json (줄인 id → 원래 id, 모델 버전, 쪼개기 여부. 브리지와 평가가 씀)
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus import examples, mbpp
from eval_bin import make_prepare, run
from log_setup import setup
from npmodel import MAGIC, NpModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bridge"))
from vocab_map import Vocab, token_bytes  # noqa: E402

LAYER_KEYS = ("input_layernorm.weight", "input_layernorm.bias",
              "post_attention_layernorm.weight", "post_attention_layernorm.bias",
              "attention.query_key_value.weight", "attention.query_key_value.bias",
              "attention.dense.weight", "attention.dense.bias",
              "mlp.dense_h_to_4h.weight", "mlp.dense_h_to_4h.bias",
              "mlp.dense_4h_to_h.weight", "mlp.dense_4h_to_h.bias")


def build_vocab(tok, texts: list[str]) -> list[int]:
    ids = {tok.eos_token_id}
    for t in texts:
        ids |= set(tok(t)["input_ids"])
    return sorted(ids)


def write_bin(path: Path, name: str, sd: dict, cfg, rows, strs: list[bytes], eos: int) -> str:
    """model.bin을 쓰고 모델 버전(<체크포인트 이름>-<내용 sha256 앞 8자리>)을 돌려준다.
    rows: 임베딩에서 남길 행 (None이면 전부, 이미 줄인 모델). strs: 줄인 id마다 토큰 바이트.
    버전은 파일 끝에 써 두어 보드가 읽어 보고한다. 가중치나 어휘가 바뀌면 버전도 자동으로 바뀐다."""
    def f32(k, sel=None):
        a = sd[k].float().numpy()
        return (a if sel is None else a[sel]).astype("<f4").tobytes()

    D, H = cfg.hidden_size, cfg.num_attention_heads
    rot = int(D // H * cfg.rope_parameters["partial_rotary_factor"])
    # 펌웨어·npmodel에 고정한 값과 같은지
    assert cfg.rope_parameters["rope_theta"] == 10000 and cfg.layer_norm_eps == 1e-5 and cfg.use_parallel_residual
    assert all(len(s) < 256 for s in strs)
    parts = [struct.pack("<8i", MAGIC, D, cfg.num_hidden_layers, H, cfg.intermediate_size, len(strs), rot, eos),
             f32("gpt_neox.embed_in.weight", rows)]
    for i in range(cfg.num_hidden_layers):
        parts += [f32(f"gpt_neox.layers.{i}.{k}") for k in LAYER_KEYS]
    parts += [f32("gpt_neox.final_layer_norm.weight"), f32("gpt_neox.final_layer_norm.bias"),
              f32("embed_out.weight", rows)]
    parts += [struct.pack("<B", len(s)) + s for s in strs]
    body = b"".join(parts)
    version = f"{name}-{hashlib.sha256(body).hexdigest()[:8]}"
    path.write_bytes(body + struct.pack("<B", len(version)) + version.encode())
    return version


def main() -> int:
    ck = Path(sys.argv[1] if len(sys.argv) > 1 else "ckpt/pythia-14m-general-k768")
    # 두 번째 인자: 출력 폴더 (비교 실험용, 보드용 firmware/model.bin·bridge/vocab.json을 덮어쓰지 않게)
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    log = setup("export")
    tok = AutoTokenizer.from_pretrained(ck)
    hf = AutoModelForCausalLM.from_pretrained(ck, dtype=torch.float32).eval()
    cfg = hf.config
    shrunk = (ck / "keep.json").exists()
    keep = json.loads((ck / "keep.json").read_text(encoding="utf-8"))["keep"] if shrunk else build_vocab(tok, examples())
    v = Vocab(keep, tok.convert_ids_to_tokens) if shrunk else None
    remap = {o: n for n, o in enumerate(keep)}
    strs = [token_bytes(tok.convert_ids_to_tokens(o)) for o in keep]
    eos = remap[tok.eos_token_id]

    out = out_dir / "model.bin" if out_dir else ROOT / "firmware" / "model.bin"
    sd = load_file(ck / "model.safetensors")  # 저장된 이름 그대로 (hf.state_dict()에는 embed_out.weight 키가 없다)
    version = write_bin(out, ck.name, sd, cfg, None if shrunk else keep, strs, eos)
    vocab = out_dir / "vocab.json" if out_dir else ROOT / "bridge" / "vocab.json"
    # 브리지가 보드의 model_version과 비교해 어휘 표가 어긋났는지 확인한다
    vocab.write_text(json.dumps({"model_version": version, "fallback": shrunk, "keep": keep, "eos": eos}),
                     encoding="utf-8")
    log.info("모델 버전 %s, 어휘 %d (%s). 저장: %s (%d 바이트), %s", version, len(keep),
             "학습 전에 줄인 모델, 쪼개기" if shrunk else "전체 어휘에서 잘라냄, 버리기", out, out.stat().st_size, vocab)

    # 1) torch와 비교: 예시 하나를 통째로 넣은 로짓(줄인 어휘 열) vs model.bin을 읽은 numpy 한 토큰씩
    m = NpModel(out)
    orig = tok(examples()[0])["input_ids"]
    small = v.encode_ids(orig)[0] if shrunk else [remap[t] for t in orig]
    with torch.no_grad():
        logits = hf(torch.tensor([small if shrunk else orig])).logits[0]
        ref = (logits if shrunk else logits[:, keep]).numpy()
    m.reset()
    got = np.stack([m.step(t) for t in small])
    diff = float(np.abs(ref - got).max())
    same = bool((ref.argmax(1) == got.argmax(1)).all())
    log.info("torch 대비 로짓 최대 오차 %.2e, argmax 일치 %s (%d토큰)", diff, same, len(small))
    assert diff < 1e-3 and same

    # 2) eval_bin.py와 같은 평가 (전체 평가는 eval_bin.py로)
    prepare = make_prepare(Tokenizer.from_pretrained("EleutherAI/pythia-14m"), json.loads(vocab.read_text("utf-8")))
    for name, texts in (("examples", examples()), ("heldout 앞 20문제", mbpp("heldout")[:20])):
        r = run(m, prepare, texts)
        log.info("%s: %d줄, 정확 일치 %.3f, 문법 통과 %.3f %s, 입력 토큰 버림·쪼갬 %.1f%%",
                 name, r["lines"], r["exact"], r["syntax_pass"], dict(r["syntax"]), r["lost_pct"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
