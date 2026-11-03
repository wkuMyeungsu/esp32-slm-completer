"""모델이 Python 코드를 얼마나 이어 쓰는지 같은 입력으로 비교한다 (파인튜닝 전후 기준선).

사용: uv run baseline.py EleutherAI/pythia-14m EleutherAI/pythia-31m [로컬 체크포인트 경로 ...]
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "# print hello world\nprint(",
    "for i in range(",
    "def add(a, b):\n    return",
    "numbers = [1, 2, 3]\ntotal = 0\nfor n in numbers:\n    total",
    "a = int(input())\nb = int(input())\nprint(a +",
]
NEW_TOKENS = 24

device = "cuda" if torch.cuda.is_available() else "cpu"
for name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    print(f"\n{'=' * 20} {name} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, {device})")
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0, ids["input_ids"].shape[1]:])
        print(f"--- {p!r}\n>>> {completion!r}")
