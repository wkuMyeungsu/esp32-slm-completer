"""공개 교재·교육 과정의 모범 코드에서 쉽고 깔끔한 Python을 골라 testsets/dev.txt, test.txt를 만든다 (학습에 쓰지 않는다).
출처·라이선스는 testsets/LICENSES.md.

출처 (저자·관리자가 검수한 코드):
  AllenDowney/ThinkPython2 code/*.py   교재 Think Python 2판 예제·연습 풀이 (MIT, 파일 머리 CC BY 4.0)
  exercism/python                      개념 연습 모범 답안(.meta/exemplar.py) + 난이도 1 연습 문제 모범 답안(.meta/example.py) (MIT)
고르는 규칙 ("쉬운 코드"):
  - 코드 줄(빈 줄·여러 줄 문자열 제외) 5~40줄
  - 클래스, yield, 데코레이터, lambda 없음
  - import는 ALLOWED 모듈만 (turtle 같은 GUI, 교재 전용 swampy, os·pickle 등 제외), open()으로 외부 파일을 읽지 않음
정리: Think Python 파일 머리의 저작권 docstring과 Python 2 호환 줄(from __future__ import ...)을 지운다 (출처는 LICENSES.md).
      corpus.normalize (\\r\\n → \\n, 탭 → 공백 4칸, 줄 끝 공백), 같은 내용은 하나만.
나누기: 프로그램을 sha256 순으로 늘어놓고 번갈아 dev·test에 넣는다 (반반, 다시 만들어도 같게).

사용: uv run --no-project python -I build_testsets.py <ThinkPython2 클론> <exercism/python 클론>
"""
import ast
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # python -I(받은 저장소 폴더의 코드가 섞이지 않게)로 돌려도 corpus를 찾게

from corpus import normalize, string_lines  # noqa: E402

ALLOWED = {"random", "math", "string", "time", "datetime", "sys", "re", "collections", "bisect", "itertools"}
MIN_LINES, MAX_LINES = 5, 40


def code_lines(text: str) -> int:
    skip = string_lines(text)
    return sum(1 for i, l in enumerate(text.splitlines()) if l.strip() and i not in skip)


def check(text: str) -> str | None:
    """규칙에 맞으면 None, 아니면 빠진 이유."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except SyntaxError:
        return "문법 오류"
    n = code_lines(text)
    if not MIN_LINES <= n <= MAX_LINES:
        return "길이"
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return "클래스"
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return "yield"
        if isinstance(node, ast.Lambda):
            return "lambda"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            return "데코레이터"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            return "외부 파일"
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""] if not node.level else ["(상대)"]
        if any(m.split(".")[0] not in ALLOWED for m in mods):
            return "import"
    return None


def strip_thinkpython(text: str) -> str:
    text = re.sub(r'^"""This module contains a code example related to.*?"""\n', "", text, count=1, flags=re.S)
    return re.sub(r"^from __future__ import .*\n", "", text, flags=re.M)


def sources(tp: Path, ex: Path):
    for p in sorted((tp / "code").glob("*.py")):
        yield f"thinkpython2/code/{p.name}", strip_thinkpython(p.read_text(encoding="utf-8", errors="replace"))
    cfg = json.loads((ex / "config.json").read_text(encoding="utf-8"))
    for e in cfg["exercises"]["concept"]:
        p = ex / "exercises" / "concept" / e["slug"] / ".meta" / "exemplar.py"
        if e.get("status") != "deprecated" and p.exists():
            yield f"exercism/concept/{e['slug']}", p.read_text(encoding="utf-8")
    for e in cfg["exercises"]["practice"]:
        p = ex / "exercises" / "practice" / e["slug"] / ".meta" / "example.py"
        if e.get("status") != "deprecated" and e["difficulty"] == 1 and p.exists():
            yield f"exercism/practice/{e['slug']}", p.read_text(encoding="utf-8")


def main() -> None:
    seen, picked, dropped = set(), [], {}
    for name, raw in sources(Path(sys.argv[1]), Path(sys.argv[2])):
        text = normalize(raw)
        why = check(text)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        if text not in seen:
            seen.add(text)
            picked.append((name, text))
    picked.sort(key=lambda x: hashlib.sha256(x[0].encode()).hexdigest())
    sets = {"dev": picked[0::2], "test": picked[1::2]}
    for split, items in sets.items():
        items.sort()
        (HERE / "testsets" / f"{split}.txt").write_text("".join(f"#### {n}\n{t}" for n, t in items), encoding="utf-8")
        by = {}
        for n, _ in items:
            k = "/".join(n.split("/")[:2])
            by[k] = by.get(k, 0) + 1
        print(f"{split}: 프로그램 {len(items)}개 {by}")
    print("빠진 이유:", dict(sorted(dropped.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
