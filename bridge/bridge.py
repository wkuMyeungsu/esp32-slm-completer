"""VS Code 확장 <-> ESP32-S3 브리지.

- stdin/stdout의 JSON 한 줄을 보드 USB 시리얼로 중계한다.
- 앞부분(prefix)을 Pythia 토크나이저로 자르고 보드의 줄인 어휘 id(vocab.json)로 바꿔 보낸다. 토큰 힐링도 여기서 한다.
- 시작할 때 보드에 버전을 물어 세션 1행(sessions), 제안마다 1행(suggestions), 화면 표시·수락마다 1행(feedback)을
  logs/rows.jsonl에 바로 남기고, 10초마다 안 보낸 줄만 Supabase의 각 테이블로 보낸다.
  이 파일은 DB에 올릴 데이터라 로테이션하지 않는다 (올리기 전에 지워지면 안 된다).
- 진단 로그(보드 시간 초과, 업로드 실패, 보드의 부팅·오류 출력)는 logs/bridge.log에 남긴다 (DEBUG 이상, 1MB × 4개 로테이션).
  콘솔은 stderr에 INFO 이상 (stdout은 확장과 주고받는 통로라 쓰지 않는다). 확장 출력 채널에 보인다.
"""
import hashlib
import json
import logging
import logging.handlers
import os
import subprocess
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
from tokenizers import Tokenizer

from syntax import syntax_check  # noqa: F401  (테스트도 bridge에서 가져다 쓴다)
from vocab_map import Vocab

ESPRESSIF_VID = 0x303A
HERE = Path(__file__).parent
ROWS = HERE / "logs" / "rows.jsonl"
OFFSET = HERE / "logs" / "rows.offset"
# 테이블별 칸 (db/schema.sql과 같게). 업로드는 이 순서대로 (세션 → 제안 → 반응)
TABLES = {
    "sessions": ("session", "started_at", "bridge_version", "fw_version", "fw_build", "model_version", "vocab_size",
                 "port"),
    "suggestions": ("session", "req", "ts", "prefix_tail", "completion", "cached", "latency_ms", "gen_ms",
                    "prompt_tokens", "dropped_tokens", "fallback_tokens", "gen_tokens", "stop_reason", "heap_min", "syntax",
                    "syntax_error"),
    "feedback": ("session", "req", "kind", "ts"),
}
SESSION = uuid.uuid4().hex[:8]
INFO_TIMEOUT = 5  # 버전 요청은 바로 답한다. 예전 펌웨어는 id 없는 오류를 보내므로 오래 기다리지 않는다
_log_lock = threading.Lock()
logger = logging.getLogger("bridge")


def setup_logging(path: Path = HERE / "logs" / "bridge.log") -> None:
    path.parent.mkdir(exist_ok=True)
    logger.setLevel(logging.DEBUG)  # 1번 문은 열어 두고 핸들러 레벨로 조절
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(filename)s:%(lineno)d [%(session)s] %(message)s",
                            defaults={"session": SESSION})
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file = logging.handlers.RotatingFileHandler(path, maxBytes=1_048_576, backupCount=3, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_version() -> str | None:
    """브리지 코드 버전. 커밋 안 된 변경이 있으면 -dirty에 bridge.py 내용 해시를 붙인다
    (커밋 전에 고친 것끼리도 DB에서 구분되게)."""
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=HERE, capture_output=True, text=True,
                           timeout=5, check=True)
        version = r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if version.endswith("-dirty"):
        version += "+" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]
    return version or None


def find_port(ports=None):
    for p in list_ports.comports() if ports is None else ports:
        if p.vid == ESPRESSIF_VID:
            return p.device
    return None


def encode(tok, vocab, prefix: str) -> tuple[list[int], str, int, int, int]:
    """앞부분 → (줄인 어휘 id, 힐링 문자열, 원래 토큰 수, 어휘에 없어 버린 토큰 수, 쪼갠 토큰 수).
    토큰 힐링: 마지막 토큰을 떼고 보드가 생성 첫 토큰들을 그 문자열에 맞추게 한다 (앞부분이 공백으로 끝날 때 토큰 경계 문제).
    vocab이 Vocab(vocab.json에 fallback: true)이면 어휘에 없는 토큰을 쪼갠다 (vocab_map.py, 학습·평가와 같은 규칙).
    dict(예전 어휘, 원래 id → 줄인 id)면 버린다."""
    ids = tok.encode(prefix).ids
    total = len(ids)
    if isinstance(vocab, Vocab):
        out, heal, fallback = vocab.encode_prefix(ids)
        return out, heal, total, 0, fallback
    heal = ""
    if len(ids) > 1:
        ids, heal = ids[:-1], tok.decode(ids[-1:])
    kept = [vocab[t] for t in ids if t in vocab]
    return kept, heal, total, len(ids) - len(kept), 0


def load_tokenizer():
    """→ (토크나이저, 어휘(Vocab 또는 원래 id → 줄인 id), vocab.json의 모델 버전)"""
    vj = json.loads((HERE / "vocab.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_pretrained("EleutherAI/pythia-14m")
    vocab = Vocab(vj["keep"], tok.id_to_token) if vj.get("fallback") else {o: n for n, o in enumerate(vj["keep"])}
    return tok, vocab, vj.get("model_version")


def out(msg: dict) -> None:
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def log_row(table: str, row: dict, log: Path = ROWS) -> None:
    assert table in TABLES, table
    row = {"table": table, "session": SESSION, "ts": now(), **row}
    log.parent.mkdir(exist_ok=True)
    with _log_lock, log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush(post, log: Path = ROWS, offset: Path = OFFSET) -> int:
    """오프셋 뒤의 완성된 줄만 테이블별로 묶어 post(table, rows)로 보내고, 모두 성공하면 오프셋을 옮긴다.
    중간에 실패하면 오프셋이 그대로라 다음에 전부 다시 보낸다 (이미 들어간 행은 supabase_post가 409로 건너뜀).
    보낸 행 수를 반환."""
    if not log.exists():
        return 0
    start = int(offset.read_text()) if offset.exists() else 0
    with log.open("rb") as f:
        f.seek(start)
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1  # 쓰는 중인 마지막 줄은 다음 차례로
    if end == 0:
        return 0
    by_table = {t: [] for t in TABLES}
    for r in map(json.loads, chunk[:end].splitlines()):
        by_table[r["table"]].append({c: r.get(c) for c in TABLES[r["table"]]})  # 일괄 삽입은 행마다 키가 같아야 한다
    for table, rows in by_table.items():
        if rows:
            post(table, rows)  # 실패하면 예외 → 오프셋 그대로
    offset.write_text(str(start + end))
    return sum(map(len, by_table.values()))


def load_env(path: Path = HERE / ".env") -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _insert(table: str, rows: list[dict]) -> None:
    req = urllib.request.Request(
        os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/" + table,
        data=json.dumps(rows).encode(),
        headers={"apikey": os.environ["SUPABASE_KEY"], "Content-Type": "application/json",
                 "Prefer": "return=minimal"},
        method="POST")
    urllib.request.urlopen(req, timeout=10).close()


def supabase_post(table: str, rows: list[dict], insert=_insert) -> None:
    """anon은 insert만 가능해서 ON CONFLICT를 못 쓴다. 묶음에 이미 들어간 행이 있으면(409) 한 행씩 넣고 중복은 건너뛴다."""
    try:
        insert(table, rows)
    except urllib.error.HTTPError as e:
        if e.code != 409:
            raise
        for r in rows:
            try:
                insert(table, [r])
            except urllib.error.HTTPError as e2:
                if e2.code != 409:
                    raise


def uploader() -> None:
    while True:
        time.sleep(10)
        try:
            n = flush(supabase_post)
            if n:
                logger.debug("Supabase 업로드 %d행", n)
        except Exception:  # 네트워크·DB 오류여도 자동완성은 계속, 파일에 남아 있으니 다음에 다시 보냄
            logger.exception("Supabase 업로드 실패, 다음 차례에 다시 보냄")


def ask_board(ser, req: dict) -> dict | None:
    """보드에 요청하고 같은 id의 응답을 기다린다. 시간 초과면 None."""
    ser.write(json.dumps(req).encode() + b"\n")
    while True:
        raw = ser.readline()
        if not raw:
            return None
        try:
            # 1바이트 토큰 때문에 UTF-8 글자가 반만 올 수 있다. 깨진 바이트는 대체 문자로 (평가의 npmodel과 같은 처리)
            msg = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            # 보드의 부팅·ESP_LOG·패닉 출력. 프로토콜엔 안 쓰지만 문제 추적용으로 남긴다
            logger.debug("보드: %s", raw.decode(errors="replace").rstrip())
            continue
        if msg.get("id") == req["id"]:
            return msg
        if "error" in msg or msg.get("ready") is False:
            logger.warning("보드 오류 응답: %s", msg)
        else:
            logger.debug("보드 (다른 id): %s", msg)


def board_info(ser) -> dict:
    """보드에 펌웨어·모델 버전을 묻는다. 브리지는 부팅이 끝난 보드에 붙으므로 부팅 메시지를 기다리지 않고 직접 묻는다."""
    timeout, ser.timeout = ser.timeout, INFO_TIMEOUT
    try:
        info = ask_board(ser, {"id": 0, "info": True}) or {}  # 확장 요청 id는 1부터
    finally:
        ser.timeout = timeout
    if "fw_version" not in info:
        logger.warning("보드가 버전 요청에 답하지 않음 (버전 보고 전 펌웨어일 수 있음): %s", info or "응답 없음")
    return info


def main() -> int:
    setup_logging()
    load_env()
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
        threading.Thread(target=uploader, daemon=True).start()
    else:
        logger.warning("SUPABASE_URL/SUPABASE_KEY 없음: 기록은 %s에만 남김", ROWS)

    tok, vocab, vocab_version = load_tokenizer()
    vocab_size = len(vocab.keep) if isinstance(vocab, Vocab) else len(vocab)
    bridge_version = git_version()
    logger.info("시작: bridge %s, 어휘 %d개 (vocab.json, 모델 %s)", bridge_version, vocab_size, vocab_version)
    port = find_port()
    if not port:
        logger.error("ESP32-S3(USB VID %04X)를 찾지 못함", ESPRESSIF_VID)
        out({"error": "ESP32-S3 not found"})
        return 1
    ser = serial.Serial(timeout=30)  # 캐시 없이 긴 앞부분(130토큰)이면 보드에서 10초 가까이 걸린다
    ser.port = port
    ser.dtr = ser.rts = False  # 열 때 DTR/RTS가 켜지면 보드가 리셋된다
    try:
        ser.open()
    except serial.SerialException:
        logger.exception("%s 열기 실패 (다른 프로그램이 포트를 쓰는 중일 수 있음)", port)
        out({"error": f"cannot open {port}"})
        return 1

    info = board_info(ser)
    board_model = info.get("model_version")
    if board_model and vocab_version and board_model != vocab_version:
        # 어휘 표가 다르면 토큰 id가 엉뚱하게 매핑되어 쓸모없는 제안만 나온다
        logger.error("보드 모델 %s와 vocab.json 모델 %s가 다름. model.bin을 다시 굽거나 vocab.json을 맞출 것",
                     board_model, vocab_version)
        out({"error": f"model mismatch: board {board_model}, vocab.json {vocab_version}"})
        return 1
    log_row("sessions", {"started_at": now(), "bridge_version": bridge_version, "fw_version": info.get("fw_version"),
                         "fw_build": info.get("fw_build"), "model_version": board_model, "vocab_size": vocab_size,
                         "port": port})
    logger.info("보드 연결: %s, fw %s (%s), model %s", port, info.get("fw_version"), info.get("fw_build"), board_model)
    out({"ready": True, "port": port, "session": SESSION})

    last_prefix, last_msg = None, None
    for line in sys.stdin:
        try:
            req = json.loads(line)
            want = req["id"]
        except (ValueError, KeyError, TypeError):
            logger.warning("확장에서 잘못된 요청: %r", line[:200])
            out({"error": "bad request"})
            continue
        if req.get("type") in ("accept", "shown"):
            log_row("feedback", {"req": want, "kind": req["type"]})
            continue

        t0 = time.perf_counter()
        prefix = req.get("prefix", "")
        ids, heal, total, dropped, fallback = encode(tok, vocab, prefix)
        cached = prefix == last_prefix
        if cached:
            # VS Code는 같은 앞부분을 연달아 요청한다. 보드를 부르지 않고 직전 결과를 준다
            msg = {**last_msg, "id": want}
            logger.debug("req %s: 직전과 같은 앞부분, 캐시 결과", want)
        else:
            logger.debug("req %s: 앞부분 %d자 → %d토큰 (버림 %d, 쪼갬 %d) → %d, heal %r", want, len(prefix), total,
                         dropped, fallback, len(ids), heal)
            try:
                msg = ask_board(ser, {"id": want, "tokens": ids, "heal": heal})
            except serial.SerialException:
                logger.exception("req %s: 시리얼 오류 (보드 연결이 끊겼을 수 있음), 브리지 종료", want)
                out({"id": want, "error": "serial"})
                break
            if msg is None:
                logger.warning("req %s: 보드 응답 시간 초과 (%s초), 입력 %d토큰", want, ser.timeout, len(ids))
                out({"id": want, "error": "timeout"})
                continue
            if "error" in msg:
                logger.warning("req %s: 보드 오류 응답 %s", want, msg["error"])
                out(msg)
                continue
            last_prefix, last_msg = prefix, msg
        out(msg)
        latency = round((time.perf_counter() - t0) * 1000)  # 문법 검사 시간은 빼고 잰다
        syntax, syntax_error = syntax_check(prefix, msg.get("completion") or "")
        logger.debug("req %s: 멈춤 %s, 문법 %s %s", want, msg.get("stop"), syntax, syntax_error or "")
        log_row("suggestions", {
            "req": want, "prefix_tail": prefix[-200:], "completion": msg.get("completion"), "cached": cached,
            "latency_ms": latency, "gen_ms": 0 if cached else msg.get("gen_ms"),
            "prompt_tokens": total, "dropped_tokens": dropped, "fallback_tokens": fallback, "gen_tokens": 0 if cached else msg.get("tokens"),
            "stop_reason": msg.get("stop") or None, "heap_min": msg.get("heap_min"),
            "syntax": syntax, "syntax_error": syntax_error})
    flush_on_exit()
    return 0


def flush_on_exit() -> None:
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
        try:
            n = flush(supabase_post)
            logger.info("종료 전 업로드 %d행", n)
        except Exception:
            logger.exception("종료 전 Supabase 업로드 실패, 다음 실행 때 다시 보냄")


if __name__ == "__main__":
    sys.exit(main())
