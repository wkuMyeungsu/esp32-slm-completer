"""학습·평가 스크립트 공통 로깅. 콘솔은 INFO 이상(요약), 파일은 DEBUG 이상(전부).

파일: train/logs/<스크립트>-<YYYYmmdd-HHMMSS>.log — LOG.md 항목은 숫자와 함께 이 경로를 적는다.
핸들러는 "train" 로거 한 곳에만 붙이고, 스크립트는 train.<이름> 자식 로거를 쓴다.
"""
import logging
import sys
import time
from pathlib import Path

LOGS = Path(__file__).parent / "logs"
FILE_FMT = "%(asctime)s %(levelname)-8s %(name)s %(filename)s:%(lineno)d %(message)s"
CONSOLE_FMT = "%(asctime)s %(levelname)-8s %(message)s"


def setup(name: str) -> logging.Logger:
    root = logging.getLogger("train")
    if not root.handlers:  # 두 번 불러도 핸들러가 겹치지 않게 (같은 줄 중복 출력 방지)
        LOGS.mkdir(exist_ok=True)
        path = LOGS / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        root.setLevel(logging.DEBUG)  # 1번 문은 열어 두고 핸들러 레벨로 조절
        root.propagate = False  # 루트 로거(라이브러리 설정)로 올려보내지 않는다
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(CONSOLE_FMT, "%H:%M:%S"))
        file = logging.FileHandler(path, encoding="utf-8")
        file.setLevel(logging.DEBUG)
        file.setFormatter(logging.Formatter(FILE_FMT))
        root.addHandler(console)
        root.addHandler(file)
        root.info("로그 파일 %s", path)
        root.info("명령 %s", " ".join([Path(sys.executable).name, *sys.argv]))
    return logging.getLogger(f"train.{name}")
