"""VS Code 확장 <-> ESP32-S3 브리지.

- stdin/stdout의 JSON 한 줄을 보드 USB 시리얼로 중계한다.
- 제안(suggest)·수락(accept)을 logs/events.jsonl에 바로 남기고, 10초마다 안 보낸 줄만 Supabase로 보낸다.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import serial
from serial.tools import list_ports

ESPRESSIF_VID = 0x303A
HERE = Path(__file__).parent
LOG = HERE / "logs" / "events.jsonl"
OFFSET = HERE / "logs" / "events.offset"
COLS = ("session", "req", "event", "ts", "model_version", "prefix_tail", "completion",
        "latency_ms", "gen_ms", "tokens", "heap_min")
SESSION = uuid.uuid4().hex[:8]
_log_lock = threading.Lock()


def find_port(ports=None):
    for p in list_ports.comports() if ports is None else ports:
        if p.vid == ESPRESSIF_VID:
            return p.device
    return None


def out(msg: dict) -> None:
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def log_event(row: dict, log: Path = LOG) -> None:
    row = {"session": SESSION, "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), **row}
    log.parent.mkdir(exist_ok=True)
    with _log_lock, log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush(post, log: Path = LOG, offset: Path = OFFSET) -> int:
    """오프셋 뒤의 완성된 줄만 post(rows)로 보내고, 성공하면 오프셋을 옮긴다. 보낸 행 수를 반환."""
    if not log.exists():
        return 0
    start = int(offset.read_text()) if offset.exists() else 0
    with log.open("rb") as f:
        f.seek(start)
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1  # 쓰는 중인 마지막 줄은 다음 차례로
    if end == 0:
        return 0
    rows = [{c: r.get(c) for c in COLS} for r in map(json.loads, chunk[:end].splitlines())]
    post(rows)  # 실패하면 예외 → 오프셋 그대로, 다음에 다시 보냄
    offset.write_text(str(start + end))
    return len(rows)


def load_env(path: Path = HERE / ".env") -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _insert(rows: list[dict]) -> None:
    req = urllib.request.Request(
        os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/events",
        data=json.dumps(rows).encode(),
        headers={"apikey": os.environ["SUPABASE_KEY"], "Content-Type": "application/json",
                 "Prefer": "return=minimal"},
        method="POST")
    urllib.request.urlopen(req, timeout=10).close()


def supabase_post(rows: list[dict], insert=_insert) -> None:
    """anon은 insert만 가능해서 ON CONFLICT를 못 쓴다. 묶음에 이미 들어간 행이 있으면(409) 한 행씩 넣고 중복은 건너뛴다."""
    try:
        insert(rows)
    except urllib.error.HTTPError as e:
        if e.code != 409:
            raise
        for r in rows:
            try:
                insert([r])
            except urllib.error.HTTPError as e2:
                if e2.code != 409:
                    raise


def uploader() -> None:
    while True:
        time.sleep(10)
        try:
            flush(supabase_post)
        except Exception as e:  # 네트워크·DB 오류여도 자동완성은 계속, 파일에 남아 있으니 다음에 다시 보냄
            print(f"upload failed: {e}", file=sys.stderr, flush=True)


def main() -> int:
    load_env()
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
        threading.Thread(target=uploader, daemon=True).start()
    else:
        print("SUPABASE_URL/SUPABASE_KEY 없음: 로그는 파일에만 남김", file=sys.stderr, flush=True)

    port = find_port()
    if not port:
        out({"error": "ESP32-S3 not found"})
        return 1
    ser = serial.Serial(timeout=10)
    ser.port = port
    ser.dtr = ser.rts = False  # 열 때 DTR/RTS가 켜지면 보드가 리셋된다
    ser.open()
    out({"ready": True, "port": port, "session": SESSION})

    for line in sys.stdin:
        try:
            req = json.loads(line)
            want = req["id"]
        except (ValueError, KeyError, TypeError):
            out({"error": "bad request"})
            continue
        if req.get("type") == "accept":
            log_event({"req": want, "event": "accept"})
            continue

        t0 = time.perf_counter()
        ser.write(json.dumps({"id": want, "prefix": req.get("prefix", "")}).encode() + b"\n")
        while True:
            raw = ser.readline()
            if not raw:
                out({"id": want, "error": "timeout"})
                break
            try:
                msg = json.loads(raw)
            except ValueError:
                continue  # 부팅 로그 같은 JSON 아닌 줄은 버린다
            if msg.get("id") == want:
                out(msg)
                log_event({"req": want, "event": "suggest",
                           "prefix_tail": req.get("prefix", "")[-200:],
                           "latency_ms": round((time.perf_counter() - t0) * 1000),
                           **{k: msg.get(k) for k in ("completion", "model_version", "gen_ms", "tokens", "heap_min")}})
                break
    flush_on_exit()
    return 0


def flush_on_exit() -> None:
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
        try:
            flush(supabase_post)
        except Exception as e:
            print(f"upload failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
