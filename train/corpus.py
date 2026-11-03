"""학습·평가용 코드 모음. 의존성 없음 (CI의 eval_bin.py가 torch 없이 쓴다).

- examples(): 직접 만든 대표 예시 8개 (data/*.py)
- mbpp(split): MBPP(Austin et al. 2021, CC-BY-4.0, data_ext/mbpp.jsonl) 974문제.
  문제 설명을 첫 줄 주석으로 붙여 playground 형식('# 영어 설명' + 코드)과 맞춘다.
  나누기: heldout = task_id 411~510 (평가 전용 100문제), train = 나머지. 공식 test(11~510) 중 앞 400개도 학습에 쓴다.
"""
import json
import warnings
from pathlib import Path

HERE = Path(__file__).parent
HELDOUT = range(411, 511)


def examples() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted((HERE / "data").glob("*.py"))]


def normalize(code: str) -> str:
    """줄바꿈 \\r\\n → \\n, 탭 → 공백 4칸 (VS Code 기본), 줄 끝 공백 제거."""
    lines = [l.expandtabs(4).rstrip() for l in code.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip("\n") + "\n"


def mbpp(split: str) -> list[str]:
    """split: 'train' 또는 'heldout'. 정리한 뒤 문법이 맞는 것만 (원문에 탭·공백이 섞인 몇 개가 깨진다)."""
    out = []
    for line in (HERE / "data_ext" / "mbpp.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (r["task_id"] in HELDOUT) != (split == "heldout"):
            continue
        text = f"# {' '.join(r['text'].split())}\n" + normalize(r["code"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 정규식 문자열의 잘못된 이스케이프 경고 (오류 아님)
                compile(text, "<mbpp>", "exec")
        except SyntaxError:
            continue
        out.append(text)
    return out


def cases(text: str):
    """(앞부분, 기대하는 다음 줄). 빈 줄은 맞힐 것이 없어 뺀다."""
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        want = lines[i].rstrip("\n")
        if want.strip():
            yield "".join(lines[:i]), want


if __name__ == "__main__":
    assert normalize("def f():\r\n\treturn 1 \r\n") == "def f():\n    return 1\n"
    assert list(cases("# a\nx = 1\n\nprint(x)\n")) == [("# a\n", "x = 1"), ("# a\nx = 1\n\n", "print(x)")]
    tr, ho = mbpp("train"), mbpp("heldout")
    print(f"examples {len(examples())}, mbpp train {len(tr)}, heldout {len(ho)}")
