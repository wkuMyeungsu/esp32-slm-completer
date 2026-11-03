"""학습·평가용 코드 모음. 의존성 없음 (CI의 eval_bin.py가 torch 없이 쓴다).

- examples(): 직접 만든 대표 예시 8개 (data/*.py)
- mbpp(split): MBPP(Austin et al. 2021, CC-BY-4.0, data_ext/mbpp.jsonl) 974문제.
  문제 설명을 첫 줄 주석으로 붙여 playground 형식('# 영어 설명' + 코드)과 맞춘다.
  나누기: heldout = task_id 411~510 (평가 전용 100문제), train = 나머지. 공식 test(11~510) 중 앞 400개도 학습에 쓴다.
"""
import io
import json
import sys
import tokenize
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


def string_lines(text: str) -> set[int]:
    """여러 줄 문자열(docstring 등) 안쪽 줄 번호(0부터). 그 안의 영어 문장은 코드 자동완성 문항에서 뺀다."""
    out = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type == tokenize.STRING and t.end[0] > t.start[0]:
                out.update(range(t.start[0], t.end[0]))  # tokenize 줄 번호는 1부터: 0부터 세면 여는 줄 다음(start)부터 닫는 줄(end-1)까지
    except (tokenize.TokenError, SyntaxError):
        pass
    return out


def cases(text: str):
    """(앞부분, 기대하는 다음 줄). 빈 줄과 여러 줄 문자열 안쪽 줄은 맞힐 코드가 아니라 뺀다."""
    lines = text.splitlines(keepends=True)
    skip = string_lines(text)
    for i in range(1, len(lines)):
        want = lines[i].rstrip("\n")
        if want.strip() and i not in skip:
            yield "".join(lines[:i]), want


def mid_cases(text: str):
    """줄 중간 커서: (앞부분 + 그 줄의 앞 절반, 그 줄의 나머지). 실제로는 줄을 치다가 제안을 받는다.
    자르는 곳은 들여쓰기를 뺀 내용의 가운데에서 가장 가까운 앞쪽 공백 뒤 (없으면 가운데 글자)."""
    lines = text.splitlines(keepends=True)
    skip = string_lines(text)
    for i, line in enumerate(lines):
        if i in skip:
            continue
        body = line.rstrip("\n")
        indent = len(body) - len(body.lstrip(" "))
        content = body[indent:]
        if len(content) < 4:
            continue
        half = len(content) // 2
        cut = content.rfind(" ", 0, half) + 1 or half
        yield "".join(lines[:i]) + body[:indent + cut], body[indent + cut:]


def testset(split: str) -> list[str]:
    """초급·실용 Python (testsets/dev.txt, test.txt, build_testsets.py가 공개 CS50P 과제 풀이에서 고름, 학습에 안 씀).
    dev: 기준값·저장 시점 조정과 CI에 쓴다. test: 최종 보고에만 쓰고 조정에 쓰지 않는다."""
    out, cur = [], None
    for line in (HERE / "testsets" / f"{split}.txt").read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("#### "):
            cur = []
            out.append(cur)
        elif cur is not None:
            cur.append(line)
    return ["".join(c).strip("\n") + "\n" for c in out]


if __name__ == "__main__":
    assert normalize("def f():\r\n\treturn 1 \r\n") == "def f():\n    return 1\n"
    assert list(cases("# a\nx = 1\n\nprint(x)\n")) == [("# a\n", "x = 1"), ("# a\nx = 1\n\n", "print(x)")]
    tr, ho = mbpp("train"), mbpp("heldout")
    print(f"examples {len(examples())}, mbpp train {len(tr)}, heldout {len(ho)}")


def capped(items, k: int = 12) -> list:
    """프로그램 하나가 점수를 좌우하지 않게 문항을 최대 k개로 고르게 뽑는다 (처음·끝 포함, 같은 간격)."""
    items = list(items)
    if len(items) <= k:
        return items
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _half(key: str) -> bool:
    """이름으로 정하는 50% (다시 만들어도 같게)."""
    import hashlib
    return hashlib.sha256(key.encode()).digest()[0] % 2 == 0


def _parses(text: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(text, "<train>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def mbpp_scripts() -> list[str]:
    """MBPP 974문제 전부를 학습용으로. 첫 줄 설명 주석은 절반만 붙이고 (모든 파일이 '# Write a function'으로
    시작한다는 편향을 줄이려고), 테스트 케이스 assert f(x) == y는 print(f(x)) 호출로 바꿔 뒤에 붙인다 (스크립트형)."""
    import re
    out = []
    for line in (HERE / "data_ext" / "mbpp.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        code = reindent(normalize(r["code"]))
        head = f"# {' '.join(r['text'].split())}\n" if _half(f"mbpp-{r['task_id']}") else ""
        calls = [m.group(1).strip() for t in r["test_list"] if (m := re.match(r"\s*assert\s+(.+?)\s*==", t))]
        tail = ("\n" + "".join(f"print({c})\n" for c in calls)) if calls else ""
        text = head + code + tail
        if _parses(text):
            out.append(text)
    return out


def instructions18k() -> list[str]:
    """iamtarun/python_code_instructions_18k_alpaca (sahil2801/code_instructions_120k에서, 라이선스 표기 없음 →
    저장소에 넣지 않고 data_ext/instructions18k/에 받아 둔다)의 output 코드. 지시문은 절반만 첫 줄 주석으로.
    문법이 맞고 2~60줄, 표준 라이브러리만 쓰는 것만 (flask·numpy·sklearn 등 제외), 같은 내용은 하나만. 들여쓰기는 4칸으로."""
    import pyarrow.parquet as pq
    folder = HERE / "data_ext" / "instructions18k"
    path = next(folder.rglob("*.parquet"), None)
    if path is None:  # 처음이면 받는다 (라이선스 표기가 없어 저장소에는 넣지 않는다)
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("iamtarun/python_code_instructions_18k_alpaca",
                                    "data/train-00000-of-00001-8b6e212f3e1ece96.parquet",
                                    repo_type="dataset", local_dir=folder))
    t = pq.read_table(path, columns=["instruction", "output"]).to_pylist()
    seen, out = set(), []
    for i, r in enumerate(t):
        code = r["output"].strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[-1].rsplit("```", 1)[0]
        code = reindent(normalize(code))
        n = len([l for l in code.splitlines() if l.strip()])
        if not 2 <= n <= 60 or code in seen or not _parses(code) or not stdlib_only(code):
            continue
        seen.add(code)
        head = f"# {' '.join(r['instruction'].split())}\n" if _half(f"i18k-{i}") and len(r["instruction"]) < 200 else ""
        out.append(head + code)
    return out


def decontaminate(texts: list[str]) -> tuple[list[str], int]:
    """테스트 세트(dev·test)의 20자 이상 줄이 들어 있는 학습 코드를 뺀다. (남은 것, 뺀 수)."""
    test_lines = {l.strip() for s in ("dev", "test") for p in testset(s) for l in p.splitlines() if len(l.strip()) >= 20}
    keep = [t for t in texts if not any(l.strip() in test_lines for l in t.splitlines())]
    return keep, len(texts) - len(keep)


def reindent(code: str) -> str:
    """블록 깊이(tokenize의 INDENT·DEDENT)를 읽어 논리 줄의 들여쓰기를 4칸 단위로 다시 맞춘다
    (MBPP·18k에는 1칸·2칸 들여쓰기가 섞여 있다). 괄호 안 이어지는 줄과 여러 줄 문자열 안쪽은 그대로. 실패하면 원문."""
    depth, starts, new_line = 0, {}, True
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.INDENT:
                depth += 1
                continue
            if t.type == tokenize.DEDENT:
                depth -= 1
                continue
            if new_line and t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.ENDMARKER):
                starts[t.start[0]] = depth
                new_line = False
            if t.type == tokenize.NEWLINE:
                new_line = True
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    lines = code.splitlines(keepends=True)
    for n, d in starts.items():
        lines[n - 1] = " " * (4 * d) + lines[n - 1].lstrip(" ")
    out = "".join(lines)
    return out if _parses(out) else code


def stdlib_only(code: str) -> bool:
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
            [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        if any(m.split(".")[0] not in sys.stdlib_module_names for m in mods):
            return False
    return True
