"""보드에 학습 예시의 앞부분을 보내 다음 줄 정확도와 속도를 잰다 (export.py의 numpy 결과와 비교용).

사용: uv run board_check.py [COM9] [--n 개수]
"""
import argparse
import json
import time
from pathlib import Path

import serial
from serial.tools import list_ports
from tokenizers import Tokenizer

from corpus import cases
from corpus import examples as load_examples
from eval_bin import make_prepare
from log_setup import setup



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=0, help="앞에서 몇 줄만 (0이면 전부)")
    a = ap.parse_args()
    log = setup("board_check")
    port =a.port or next(p.device for p in list_ports.comports() if p.vid == 0x303A)
    vocab_json = json.loads((Path(__file__).parent.parent / "bridge" / "vocab.json").read_text(encoding="utf-8"))
    prepare = make_prepare(Tokenizer.from_pretrained("EleutherAI/pythia-14m"), vocab_json)  # 브리지와 같은 토큰화

    ser = serial.Serial(timeout=120)
    ser.port = port
    ser.dtr = ser.rts = False  # 열 때 보드가 리셋되지 않게
    ser.open()
    todo = [c for t in load_examples() for c in cases(t)]
    todo = todo[:a.n] if a.n else todo
    hit, gen_ms, made, heap = 0, 0, 0, None
    for i, (prefix, want) in enumerate(todo):
        ids, heal, _, _ = prepare(prefix)
        ser.write(json.dumps({"id": i, "tokens": ids, "heal": heal}).encode() + b"\n")
        t0 = time.perf_counter()
        while True:
            raw = ser.readline()
            if not raw:
                log.error("%d: 보드 응답 시간 초과 (120초)", i)
                raise SystemExit(1)
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                log.debug("보드 (JSON 아님): %r", raw)
                continue
            if msg.get("id") == i:
                break
            log.warning("보드 (다른 id): %s", msg)
        got = msg["completion"].split("\n")[0]
        hit += got == want
        gen_ms += msg["gen_ms"]
        made += msg["tokens"]
        heap = msg["heap_min"]
        log.info("%s 입력 %3d토큰, 생성 %2d토큰, %6d ms, 왕복 %5.1f s  %r%s", "O" if got == want else "X", len(ids),
                 msg["tokens"], msg["gen_ms"], time.perf_counter() - t0, got, "" if got == want else f" (기대 {want!r})")
    n = len(todo)
    log.info("다음 줄 정확도 %.3f (%d/%d), 평균 생성 시간 %.0f ms, 토큰당 %.0f ms (앞부분 처리 포함), 최소 남은 힙 %s",
             hit / n, hit, n, gen_ms / n, gen_ms / max(made, 1), heap)


if __name__ == "__main__":
    main()
