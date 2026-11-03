"""보드에 굽는 model.bin을 numpy로 돌려 평가한다. 기준 미만이면 실패 (CI 평가). torch 없이 numpy와 tokenizers만 쓴다.

세트 (줄 시작: 앞부분을 줄 경계마다 잘라 다음 한 줄, 줄 중간: 줄을 절반쯤 친 상태에서 그 줄의 나머지):
  examples      직접 만든 대표 예시 8개. 학습에 8번씩 넣어서 외우기 회귀 확인일 뿐, 정확도 근거가 아니다
  dev, dev_mid  CS50P 과제 풀이 45개 (testsets/dev.txt, build_testsets.py, 학습에 안 씀). 기준값 조정·저장 시점·CI 기준에 쓴다
  test, test_mid CS50P 과제 풀이 50개 (다른 과제). --final일 때만 잰다. 조정에 쓰지 않는다 (최종 보고용)
  dev·test는 프로그램 하나가 점수를 좌우하지 않게 프로그램당 최대 12문항 (corpus.capped)
  heldout       MBPP 100문제 (학습에 안 씀). 2026-11-03 저장 시점·반복·확신 기준을 고를 때 이미 썼으므로 공개 비교용으로만
지표: 정확 일치, 앞부분 맞음(빈 제안이 아니고 실제 나머지의 앞부분과 맞음), 줄당 맞는 글자, 빈 제안 비율, 문법 통과.
문법 판정은 브리지와 같은 bridge/syntax.py. 토큰화도 브리지와 같은 bridge/vocab_map.py
(vocab.json에 fallback이 없는 예전 어휘는 어휘에 없는 토큰을 버린다).

사용: uv run --no-project --with numpy --with tokenizers python train/eval_bin.py --min-examples 0.95 --min-dev-syntax 0.8 [--final]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bridge"))

from corpus import capped, cases, examples, mbpp, mid_cases, testset  # noqa: E402
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


def run(m: NpModel, prepare, texts: list[str], repeat: int | None = None, case_fn=cases) -> dict:
    """repeat: 반복이면 멈추는 기준 (None이면 npmodel 기본값 REPEAT, 0이면 끔).
    prefix_ok: 빈 제안이 아니고 실제 줄의 앞부분과 맞는 비율 (제안이 짧아도 맞으면 맞음)."""
    r = {"lines": 0, "hit": 0, "prefix_ok": 0, "empty": 0, "chars": 0, "useful": 0, "syntax": Counter(), "stop": Counter(),
         "tokens": 0, "lost": 0, "misses": []}
    kw = {} if repeat is None else {"repeat": repeat}
    for text in texts:
        m.reset()  # 프로그램마다 캐시를 비운다 (같은 프로그램 안에서는 앞부분을 이어 쓴다)
        for prefix, want in case_fn(text):
            ids, heal, n, lost = prepare(prefix)
            got = m.generate(ids, heal, **kw).split("\n")[0]
            r["lines"] += 1
            r["hit"] += got == want
            r["empty"] += got == ""
            good = got != "" and want.startswith(got)
            r["prefix_ok"] += good
            r["useful"] += len(got) if good else 0
            r["chars"] += len(got)
            r["stop"][m.last_stop] += 1
            r["syntax"][syntax_check(prefix, got)[0]] += 1
            r["tokens"] += n
            r["lost"] += lost
            if got != want:
                r["misses"].append((prefix.splitlines()[-1] if prefix.strip() else "", want, got))
    judged = sum(r["syntax"][k] for k in ("ok", "incomplete", "error"))
    r["exact"] = r["hit"] / r["lines"]
    r["prefix_ok_pct"] = r["prefix_ok"] / max(r["lines"] - r["empty"], 1)
    r["empty_pct"] = r["empty"] / r["lines"]
    r["avg_chars"] = r["chars"] / max(r["lines"] - r["empty"], 1)
    r["useful_per_line"] = r["useful"] / r["lines"]  # 사용자가 맞는 제안만 받는다고 할 때 줄당 아껴 주는 글자
    r["syntax_pass"] = (r["syntax"]["ok"] + r["syntax"]["incomplete"]) / judged if judged else 0.0
    r["lost_pct"] = 100 * r["lost"] / r["tokens"]
    return r


def main() -> int:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=ROOT / "firmware" / "model.bin")
    ap.add_argument("--torch", type=Path, default=None, help="학습 체크포인트 폴더를 GPU로 평가 (model.bin 대신, train 환경에서)")
    ap.add_argument("--vocab", type=Path, default=ROOT / "bridge" / "vocab.json")
    ap.add_argument("--min-examples", type=float, default=0.0, help="예시 다음 줄 정확도가 이보다 낮으면 실패")
    ap.add_argument("--min-dev-syntax", type=float, default=0.0, help="dev(줄 시작) 문법 통과율이 이보다 낮으면 실패")
    ap.add_argument("--min-dev-mid-prefix", type=float, default=0.0, help="dev_mid 앞부분 맞음 비율이 이보다 낮으면 실패")
    ap.add_argument("--heldout", type=int, default=0, help="MBPP heldout 문제 수 (기본 0: mix 학습에는 MBPP 전부가 들어가 평가로 못 씀)")
    ap.add_argument("--final", action="store_true", help="test 세트도 잰다 (모델·설정을 확정한 뒤에만)")
    a = ap.parse_args()
    log = setup("eval_bin")
    tok = Tokenizer.from_pretrained("EleutherAI/pythia-14m")
    vocab_json = json.loads(a.vocab.read_text(encoding="utf-8"))
    if a.torch:  # 학습 체크포인트를 GPU로 (빠름, 생성 규칙은 같은 함수). 어휘는 체크포인트의 keep.json
        from torchmodel import TorchModel
        m = TorchModel(a.torch)
        vocab_json = {"keep": json.loads((a.torch / "keep.json").read_text(encoding="utf-8"))["keep"], "fallback": True}
    else:
        m = NpModel(a.bin)
        assert vocab_json.get("model_version") in (None, m.version), "model.bin과 vocab.json의 모델 버전이 다르다"
    assert len(m.strs) == len(vocab_json["keep"]), "모델과 어휘 표의 크기가 다르다"
    log.info("모델 %s, 어휘 %d, %s", m.version, len(m.strs), "쪼개기" if vocab_json.get("fallback") else "버리기")
    prepare = make_prepare(tok, vocab_json)

    fail = False
    sets = [("examples", examples(), cases, a.min_examples, "exact"),
            ("dev", testset("dev"), lambda t: capped(cases(t)), a.min_dev_syntax, "syntax_pass"),
            ("dev_mid", testset("dev"), lambda t: capped(mid_cases(t)), a.min_dev_mid_prefix, "prefix_ok_pct")]
    if a.final:
        sets += [("test", testset("test"), lambda t: capped(cases(t)), 0.0, "exact"),
                 ("test_mid", testset("test"), lambda t: capped(mid_cases(t)), 0.0, "exact")]
    if a.heldout:
        sets.append(("heldout", mbpp("heldout")[:a.heldout], cases, 0.0, "exact"))
    for name, texts, fn, gate, key in sets:
        r = run(m, prepare, texts, case_fn=fn)
        log.info("%s: %d줄, 정확 일치 %.3f (%d), 앞부분 맞음 %.3f, 줄당 맞는 글자 %.2f, 빈 제안 %.3f, 문법 통과 %.3f %s, "
                 "멈춤 %s, 입력 토큰 %d 중 버림·쪼갬 %.1f%%",
                 name, r["lines"], r["exact"], r["hit"], r["prefix_ok_pct"], r["useful_per_line"], r["empty_pct"],
                 r["syntax_pass"], dict(r["syntax"]), dict(r["stop"]), r["tokens"], r["lost_pct"])
        for last, want, got in r["misses"][:15]:
            log.debug("  %s 틀림: 앞줄 %r, 기대 %r, 생성 %r", name, last, want, got)
        if r[key] < gate:
            log.error("%s: %s %.3f < 기준 %s", name, key, r[key], gate)
            fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
