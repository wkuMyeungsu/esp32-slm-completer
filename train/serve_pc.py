"""PC(GPU)에서 모델을 돌려 VS Code 확장에 완성을 준다. 브리지와 같은 stdin/stdout JSON 한 줄 형식.

보드에 올리기 전에 모델을 VS Code에서 직접 써 보는 용도. DB 로그는 남기지 않는다.
사용: uv run serve_pc.py EleutherAI/pythia-31m
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 32


def cut_line(text: str) -> str:
    """완성은 한 줄까지. 시작이 줄바꿈(예: `:` 다음)이면 다음 줄 하나까지 준다."""
    head = len(text) - len(text.lstrip("\n"))
    end = text.find("\n", head)
    return text if end == -1 else text[:end]


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-31m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    print(json.dumps({"ready": True, "port": f"PC:{name}:{device}", "session": "pc"}), flush=True)

    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("type") == "accept" or "id" not in req:
            continue
        t0 = time.perf_counter()
        ids = tok(req.get("prefix", ""), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        new = out[0, ids["input_ids"].shape[1]:]
        print(json.dumps({"id": req["id"], "completion": cut_line(tok.decode(new)),
                          "model_version": name, "gen_ms": round((time.perf_counter() - t0) * 1000),
                          "tokens": len(new)}), flush=True)
    return 0


if __name__ == "__main__":
    assert cut_line("10):\n    print(i)") == "10):"
    assert cut_line("\n    print(i)\nfoo") == "\n    print(i)"
    sys.exit(main())
