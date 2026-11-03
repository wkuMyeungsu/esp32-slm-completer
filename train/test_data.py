"""data/의 대표 예시가 실제로 실행되고 기대한 출력을 내는지 확인한다. 표준 라이브러리만 쓴다 (CI에서 torch 없이 실행)."""
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"

# 파일 이름: (stdin, 기대 출력)
EXPECTED = {
    "01_hello.py": ("", "Hello, World!\n"),
    "02_variables.py": ("", "My name is Alice.\nI am 20 years old.\n"),
    "03_grade.py": ("", "Score 85 is grade B.\n"),
    "04_times_table.py": ("", "".join(f"2 x {i} = {2 * i}\n" for i in range(1, 10))),
    "05_while_sum.py": ("", "Sum: 55\n"),
    "06_list_stats.py": ("", "Sum: 26\nAverage: 5.2\nMax: 9\n"),
    "07_factorial.py": ("", "120\n"),
    "08_calculator.py": ("6\n*\n7\n", "First number: Operator (+, -, *, /): Second number: 42\n"),
}


def test_every_example_has_expected_output():
    assert sorted(p.name for p in DATA.glob("*.py")) == sorted(EXPECTED)


def test_examples_run_and_print_expected():
    for name, (stdin, want) in EXPECTED.items():
        run = subprocess.run([sys.executable, str(DATA / name)], input=stdin,
                             capture_output=True, text=True, timeout=10)
        assert run.returncode == 0, (name, run.stderr)
        assert run.stdout == want, name


def test_calculator_divide_by_zero():
    run = subprocess.run([sys.executable, str(DATA / "08_calculator.py")], input="1\n/\n0\n",
                         capture_output=True, text=True, timeout=10)
    assert run.stdout.endswith("Cannot divide by zero\n")


def test_first_lines_are_unique_comments():
    firsts = [p.read_text(encoding="utf-8").splitlines()[0] for p in DATA.glob("*.py")]
    assert all(f.startswith("# ") for f in firsts)
    assert len(set(firsts)) == len(firsts)  # 첫 줄로 어느 예시인지 구분돼야 한다
