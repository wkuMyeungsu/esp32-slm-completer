import json
from types import SimpleNamespace as P
from urllib.error import HTTPError

import pytest

from bridge import TABLES, find_port, flush, log_row, supabase_post


def test_find_port_picks_espressif():
    ports = [P(vid=None, device="COM1"), P(vid=0x10C4, device="COM3"), P(vid=0x303A, device="COM9")]
    assert find_port(ports) == "COM9"


def test_find_port_none():
    assert find_port([P(vid=None, device="COM1")]) is None


def test_flush_groups_by_table_and_keeps_rows_on_failure(tmp_path):
    log, off = tmp_path / "r.jsonl", tmp_path / "r.offset"
    sent = []
    post = lambda table, rows: sent.extend((table, r) for r in rows)
    log_row("feedback", {"req": 1, "kind": "accept"}, log)
    log_row("suggestions", {"req": 1, "completion": "x", "cached": False}, log)
    log_row("sessions", {"started_at": "t", "fw_version": "abc"}, log)

    assert flush(post, log, off) == 3
    assert [t for t, _ in sent] == ["sessions", "suggestions", "feedback"]  # 세션 → 제안 → 반응 순서
    assert all(set(r) == set(TABLES[t]) for t, r in sent)  # 일괄 삽입은 모든 행의 키가 같아야 한다
    assert sent[1][1]["stop_reason"] is None and "table" not in sent[1][1]
    assert flush(post, log, off) == 0  # 이미 보낸 줄은 다시 안 보냄

    log_row("suggestions", {"req": 2, "cached": True}, log)

    def fail(table, rows):
        raise OSError("offline")

    with pytest.raises(OSError):
        flush(fail, log, off)
    assert flush(post, log, off) == 1  # 실패한 줄은 다음에 보냄
    assert sent[-1][1]["req"] == 2


def test_supabase_post_skips_duplicates_one_by_one():
    db = {1}  # req 1은 이미 들어가 있음

    def insert(table, rows):
        assert table == "suggestions"
        if any(r["req"] in db for r in rows):
            raise HTTPError("u", 409, "conflict", None, None)
        db.update(r["req"] for r in rows)

    supabase_post("suggestions", [{"req": 1}, {"req": 2}], insert)
    assert db == {1, 2}

    def down(table, rows):
        raise HTTPError("u", 503, "unavailable", None, None)

    with pytest.raises(HTTPError):  # 409가 아닌 실패는 그대로 올려서 오프셋이 안 움직이게
        supabase_post("suggestions", [{"req": 3}], down)


def test_flush_skips_partial_last_line(tmp_path):
    log, off = tmp_path / "r.jsonl", tmp_path / "r.offset"
    first = b'{"table": "feedback", "session": "s", "req": 1, "kind": "accept", "ts": "t"}\n'
    log.write_bytes(first + b'{"table": "feedback", "req": 2, "ki')
    assert flush(lambda t, r: None, log, off) == 1
    assert int(off.read_text()) == len(first)


SRC = "def add(a, b):\n    return a + b\n\n\n"


@pytest.fixture(scope="module")
def tok():
    from bridge import load_tokenizer
    t, _, version = load_tokenizer()
    assert version  # export.py가 vocab.json에 모델 버전을 넣는다
    return t


def test_encode_drop_mode_heals_and_counts_dropped(tok):
    from bridge import encode
    remap = {o: n for n, o in enumerate(sorted(set(tok.encode("def add(a, b):\n    return").ids)))}  # 예전 어휘
    ids, heal, total, dropped, fallback = encode(tok, remap, SRC)
    assert heal == "\n\n\n"  # 학습 때는 '\n\n' + '\n'으로 끊기던 끝 공백을 떼어 보드가 다시 맞춘다
    assert total == len(tok.encode(SRC).ids) and fallback == 0
    assert dropped == total - 1 - len(ids) > 0  # 힐링으로 뗀 1개는 버린 것이 아니다
    assert encode(tok, remap, "") == ([], "", 0, 0, 0)


def test_encode_fallback_mode_keeps_every_byte(tok):
    from bridge import encode
    from vocab_map import Vocab, token_bytes
    small = [i for i in range(tok.get_vocab_size()) if len(token_bytes(tok.id_to_token(i))) == 1]  # 1바이트 토큰만
    v = Vocab(sorted(set(small) | {tok.encode("def").ids[0]}), tok.id_to_token)
    src = "import numpy as np\nx = np.zeros(3)  # 한글\n"
    ids, heal, total, dropped, fallback = encode(tok, v, src)
    assert dropped == 0 and fallback > 0
    assert b"".join(v.bytes[i] for i in ids) + heal.encode() == src.encode()  # 쪼개도 입력이 그대로 남는다


class Ser:
    def __init__(self, lines):
        self.lines, self.sent, self.timeout = list(lines), b"", 30

    def write(self, b):
        self.sent += b

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


def test_ask_board_skips_other_lines_and_times_out():
    from bridge import ask_board
    ser = Ser([b"I (123) boot log\n", b'{"id": 1, "completion": "old"}\n', b'{"id": 2, "completion": "x"}\n'])
    assert ask_board(ser, {"id": 2, "tokens": [5, 6], "heal": "("})["completion"] == "x"
    assert json.loads(ser.sent) == {"id": 2, "tokens": [5, 6], "heal": "("}
    assert ask_board(Ser([]), {"id": 3}) is None


def test_board_info_restores_timeout_and_tolerates_old_firmware():
    from bridge import board_info
    ser = Ser([b'{"id": 0, "fw_version": "589d8d3-dirty", "fw_build": "a1b2c3d4", "model_version": "m-1", "vocab": 152}\n'])
    assert board_info(ser)["fw_version"] == "589d8d3-dirty"
    assert ser.timeout == 30 and json.loads(ser.sent) == {"id": 0, "info": True}
    old = Ser([b'{"error": "bad request"}\n'])  # 버전 보고 전 펌웨어: id 없는 오류 → 시간 초과
    assert board_info(old) == {} and old.timeout == 30


@pytest.mark.parametrize("prefix, completion, want", [
    ('# hello world\n', 'print("Hello, World!")', "ok"),
    ('# t\n', 'for i in range(1, 10):', "incomplete"),  # 블록이 이어질 차례
    ('total = ', 'sum(', "incomplete"),  # 괄호가 아직 안 닫힘
    ('# t\n', 'for i in range(1, 10);', "error"),
    ('for i in range(3):\n    ', 'af" = " = ")', "error"),  # 실사용에서 나온 반복 출력
    ('x = (\n', 'print(1)', "incomplete"),
    ('print("a"\nx = 1\n', 'y = 2', "prefix_error"),  # 완성 전 줄이 이미 오류 → 모델 탓 아님
    # 실사용: 커서 줄이 이미 깨진 상태(따옴표 안 닫힘)에서 붙인 완성 → 같은 오류라 모델 탓 아님
    ('# hello world\nprint("Hello, World!") in {name}.") ', 'of', "prefix_error"),
    ('# hello world\nprint("Hello, World!")  ', '  )', "error"),  # 완성이 새로 만든 오류
    ('x = "abc', '"', "ok"),  # 앞부분만 보면 오류여도 완성이 고치면 ok
    ('# hello world\nprint("Hello, World!") in', '', None),  # 빈 완성은 판단 안 함
    ('a' * 1024, 'x', None),  # 앞부분이 잘렸으면 판단 안 함
])
def test_syntax_check(prefix, completion, want):
    from bridge import syntax_check
    status, msg = syntax_check(prefix, completion)
    assert status == want
    assert (msg is not None) == (status in ("error", "prefix_error"))
