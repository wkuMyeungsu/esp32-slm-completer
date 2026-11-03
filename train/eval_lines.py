"""예시를 줄 경계마다 잘라 앞부분을 주고, 다음 한 줄을 정확히 맞히는지 본다 (외워서 되살리기 평가).

사용: uv run eval_lines.py EleutherAI/pythia-14m ckpt/pythia-14m-ft
"""
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"


def load_examples() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(DATA.glob("*.py"))]


def cases(text: str):
    """(앞부분, 기대하는 다음 줄). 빈 줄은 맞힐 것이 없어 뺀다."""
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        want = lines[i].rstrip("\n")
        if want.strip():
            yield "".join(lines[:i]), want


def evaluate(model, tok, texts: list[str], device: str, max_new_tokens: int = 32) -> dict:
    import torch  # cases·load_examples는 torch 없이 쓰게 (CI의 eval_bin.py)
    torch.set_grad_enabled(False)
    hit = total = 0
    misses = []
    for text in texts:
        for prefix, want in cases(text):
            ids = tok(prefix, return_tensors="pt").to(device)
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            got = tok.decode(out[0, ids["input_ids"].shape[1]:]).split("\n")[0]
            total += 1
            if got == want:
                hit += 1
            else:
                misses.append((want, got))
    return {"line_exact": round(hit / total, 3), "hit": hit, "lines": total, "misses": misses}


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from log_setup import setup
    log = setup("eval_lines")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = load_examples()
    for name in sys.argv[1:]:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        r = evaluate(model, tok, texts, device)
        log.info("%s: 다음 줄 정확도 %s (%d/%d)", name, r["line_exact"], r["hit"], r["lines"])
        for want, got in r["misses"]:
            log.info("  틀림: 기대 %r, 생성 %r", want, got)
    return 0


if __name__ == "__main__":
    assert list(cases("# a\nx = 1\n\nprint(x)\n")) == [("# a\n", "x = 1"), ("# a\nx = 1\n\n", "print(x)")]
    sys.exit(main())
