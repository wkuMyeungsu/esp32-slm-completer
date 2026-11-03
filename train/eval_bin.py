"""보드에 굽는 model.bin을 numpy로 돌려 평가한다. 기준 미만이면 실패 (CI 평가). torch 없이 numpy와 tokenizers만 쓴다.

세트 두 개 (앞부분을 줄 경계마다 잘라 다음 한 줄을 생성):
  examples  직접 만든 대표 예시 8개 (학습에 씀) → 다음 줄 정확도 (원래 하던 것이 망가지지 않았나)
  heldout   MBPP 평가 전용 100문제 (학습에 안 씀) → 문법 통과율·정확 일치율 (처음 보는 코드에서)
문법 판정은 브리지와 같은 bridge/syntax.py. 토큰화도 브리지와 같은 bridge/vocab_map.py
(vocab.json에 fallback이 없는 예전 어휘는 어휘에 없는 토큰을 버린다).

사용: uv run --no-project --with numpy --with tokenizers python train/eval_bin.py --min-examples 0.9 --min-heldout-syntax 0.5
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bridge"))

from corpus import cases, examples, mbpp  # noqa: E402
from log_setup import setup  # noqa: E402
from npmodel import NpModel, encode  # noqa: E402
from syntax import syntax_check  # noqa: E402
from vocab_map import Vocab  # noqa: E402


def make_prepare(tok, vocab_json: dict):
    """앞부분 → (줄인 id, 힐링 문자열, 원래 토큰 수, 버리거나 쪼갠 토큰 수). 브리지와 같은 규칙."""
    keep = vocab_json["keep"]
    if vocab_json.get("fallback"):
        v = Vocab(keep, tok.id_to_token)

        def prepare(prefix):
            ids = tok.encode(prefix).ids
            out, heal, n = v.encode_prefix(ids)
            return out, heal, len(ids), n
    else:
        remap = {o: n for n, o in enumerate(keep)}

        def prepare(prefix):
            ids = tok.encode(prefix).ids
            out, heal = encode(ids, tok.decode, remap)
            return out, heal, len(ids), len(ids) - (1 if heal else 0) - len(out)
    return prepare


def run(m: NpModel, prepare, texts: list[str]) -> dict:
    r = {"lines": 0, "hit": 0, "syntax": Counter(), "tokens": 0, "lost": 0, "misses": []}
    for text in texts:
        m.reset()  # 프로그램마다 캐시를 비운다 (같은 프로그램 안에서는 앞부분을 이어 쓴다)
        for prefix, want in cases(text):
            ids, heal, n, lost = prepare(prefix)
            got = m.generate(ids, heal).split("\n")[0]
            r["lines"] += 1
            r["hit"] += got == want
            r["syntax"][syntax_check(prefix, got)[0]] += 1
            r["tokens"] += n
            r["lost"] += lost
            if got != want:
                r["misses"].append((prefix.splitlines()[-1] if prefix.strip() else "", want, got))
    judged = sum(r["syntax"][k] for k in ("ok", "incomplete", "error"))
    r["exact"] = r["hit"] / r["lines"]
    r["syntax_pass"] = (r["syntax"]["ok"] + r["syntax"]["incomplete"]) / judged if judged else 0.0
    r["lost_pct"] = 100 * r["lost"] / r["tokens"]
    return r


def main() -> int:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=ROOT / "firmware" / "model.bin")
    ap.add_argument("--vocab", type=Path, default=ROOT / "bridge" / "vocab.json")
    ap.add_argument("--min-examples", type=float, default=0.0, help="예시 다음 줄 정확도가 이보다 낮으면 실패")
    ap.add_argument("--min-heldout-syntax", type=float, default=0.0, help="평가 세트 문법 통과율이 이보다 낮으면 실패")
    ap.add_argument("--heldout", type=int, default=100, help="평가 세트에서 쓸 문제 수")
    a = ap.parse_args()
    log = setup("eval_bin")
    tok = Tokenizer.from_pretrained("EleutherAI/pythia-14m")
    vocab_json = json.loads(a.vocab.read_text(encoding="utf-8"))
    m = NpModel(a.bin)
    assert len(m.strs) == len(vocab_json["keep"]), "model.bin과 vocab.json의 어휘 크기가 다르다"
    assert vocab_json.get("model_version") in (None, m.version), "model.bin과 vocab.json의 모델 버전이 다르다"
    log.info("모델 %s, 어휘 %d, %s", m.version, len(m.strs), "쪼개기" if vocab_json.get("fallback") else "버리기")
    prepare = make_prepare(tok, vocab_json)

    fail = False
    for name, texts, gate, key in (("examples", examples(), a.min_examples, "exact"),
                                   ("heldout", mbpp("heldout")[:a.heldout], a.min_heldout_syntax, "syntax_pass")):
        r = run(m, prepare, texts)
        log.info("%s: %d줄, 정확 일치 %.3f (%d), 문법 통과 %.3f %s, 입력 토큰 %d 중 버림·쪼갬 %.1f%%",
                 name, r["lines"], r["exact"], r["hit"], r["syntax_pass"], dict(r["syntax"]), r["tokens"], r["lost_pct"])
        for last, want, got in r["misses"][:15]:
            log.debug("  %s 틀림: 앞줄 %r, 기대 %r, 생성 %r", name, last, want, got)
        if r[key] < gate:
            log.error("%s: %s %.3f < 기준 %s", name, key, r[key], gate)
            fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
