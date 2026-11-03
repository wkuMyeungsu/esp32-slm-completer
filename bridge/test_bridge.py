from types import SimpleNamespace as P

import pytest

from bridge import COLS, find_port, flush, log_event


def test_find_port_picks_espressif():
    ports = [P(vid=None, device="COM1"), P(vid=0x10C4, device="COM3"), P(vid=0x303A, device="COM9")]
    assert find_port(ports) == "COM9"


def test_find_port_none():
    assert find_port([P(vid=None, device="COM1")]) is None


def test_flush_sends_only_new_rows_and_keeps_them_on_failure(tmp_path):
    log, off = tmp_path / "e.jsonl", tmp_path / "e.offset"
    sent = []
    log_event({"req": 1, "event": "suggest", "completion": "x"}, log)
    log_event({"req": 1, "event": "accept"}, log)

    assert flush(sent.extend, log, off) == 2
    assert all(set(r) == set(COLS) for r in sent)  # 일괄 삽입은 모든 행의 키가 같아야 한다
    assert sent[1]["event"] == "accept" and sent[1]["completion"] is None
    assert flush(sent.extend, log, off) == 0  # 이미 보낸 줄은 다시 안 보냄

    log_event({"req": 2, "event": "suggest"}, log)

    def fail(rows):
        raise OSError("offline")

    with pytest.raises(OSError):
        flush(fail, log, off)
    assert flush(sent.extend, log, off) == 1  # 실패한 줄은 다음에 보냄
    assert [r["req"] for r in sent] == [1, 1, 2]


def test_flush_skips_partial_last_line(tmp_path):
    log, off = tmp_path / "e.jsonl", tmp_path / "e.offset"
    log.write_bytes(b'{"req": 1, "event": "suggest"}\n{"req": 2, "ev')
    sent = []
    assert flush(sent.extend, log, off) == 1
    assert int(off.read_text()) == len(b'{"req": 1, "event": "suggest"}\n')
