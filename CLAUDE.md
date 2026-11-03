# esp32-slm-completer

## 공통 컨텍스트 메모리

`.agents/memory.txt`에 이 프로젝트의 공통 컨텍스트 메모리 로그가 있다. 작업을 시작하기 전에 읽고, 진행 상황이 바뀌면 한 줄씩 덧붙인다 (`YYYY-MM-DD HH:MM 내용`). 이 폴더는 로컬 전용이라 git에 올리지 않는다 (`.gitignore`).

## 학습 기록

학습·평가의 시행착오(명령, 숫자 결과, 원인, 조치)는 `train/LOG.md`에 시간 순으로 이어 쓴다.

## 브랜치

- 기본 브랜치는 `dev` (GitHub 기본 브랜치도 `dev`).
- 작업은 `dev`에서 한다. 온전한 기능 하나가 완성됐다고 판단하기 전까지 `main`에 넣지 않는다.

## 오늘 주제 (Day20, 2026-11-03 개인 프로젝트)

- 주제는 자유. 마감 16:30, 발표 없음.
- 필수 조건 세 가지:
  1. ESP32에 모델을 넣어 사용
  2. 로그를 DB에 저장
  3. CI 구현

## 선정한 주제

ESP32-S3를 활용한 SLM 에이전트 코드 컴플리터.

- 직접 고른 Python 튜토리얼 예제(헬로월드, 반복문, 사칙연산 등)로 학습한 SLM을 ESP32-S3에 올린다.
- VS Code 확장에서 ESP32-S3가 잡히면 SLM 서버와 주고받으며, 입력하는 Python 코드를 자동완성한다.
- 보드 펌웨어는 ESP-IDF.

## 구조

| 폴더 | 역할 |
|---|---|
| `firmware/` | ESP-IDF 펌웨어. USB 시리얼로 JSON 한 줄 요청 `{"id","tokens","heal"}`을 받아 `main/model.c`로 추론하고 `{"id","completion","stop","model_version","gen_ms","tokens","heap_min"}`으로 답한다. `{"id":0,"info":true}`에는 펌웨어·모델 버전을 답한다. 가중치 `model.bin`은 `model` 파티션(0x190000)에 따로 굽는다 |
| `bridge/` | Python 브리지. Espressif USB(VID 303A) 포트를 찾아 확장과 보드 사이를 중계한다. 앞부분을 Pythia 토크나이저로 잘라 `vocab.json`의 줄인 어휘 id로 바꾸고(어휘에 없는 토큰은 `vocab_map.py`로 쪼갬) 토큰 힐링 문자열을 붙여 보낸다. 기록을 `logs/rows.jsonl`에 남긴 뒤 10초마다 Supabase 세 테이블로 보낸다. 문법 검사 `syntax.py`, 진단 로그 `logs/bridge.log` |
| `train/` | 범용 학습(`train_general.py`, 데이터 `corpus.py`: 예시 8개 + MBPP), 예시만 외우는 학습(`finetune.py`), 보드용 내보내기(`export.py` → `firmware/model.bin`, `bridge/vocab.json`), numpy 추론(`npmodel.py`, 펌웨어 `model.c`의 기준), 평가(`eval_bin.py` CI용, `board_check.py` 보드용) |
| `extension/` | VS Code 확장. Python 파일에서 커서 앞 코드를 보내고 인라인 제안으로 보여 준다. 제안을 수락하면 브리지에 `{"type":"accept","id"}`를 보낸다 |
| `db/schema.sql` | Supabase `sessions`(브리지 실행·버전)·`suggestions`(제안)·`feedback`(shown·accept) 테이블, anon은 insert만 허용, 성적표 뷰 `scorecard`·`stop_reasons` |
| `.github/workflows/ci.yml` | CI: 브리지 pytest, 학습 예시 실행, model.bin 평가(예시 0.95·heldout 문법 0.85 미만 실패), 확장 패키징·통합 테스트(실제 VS Code + 가짜 브리지), 펌웨어 ESP-IDF 빌드 |

## 펌웨어 빌드·굽기 (로컬, ESP-IDF v6.1)

저장소 경로에 한글이 있어 `idf.py build`는 cmake가 비정상 종료한다(0xC0000409). `idf.py`는 junction을 따라가므로 소용없고, 영문 junction `D:\esp\esp32-slm-completer`로 cmake·ninja를 직접 부른다.

```powershell
. C:\Espressif\tools\Microsoft.v6.1.PowerShell_profile.ps1
cmake -G Ninja -S D:\esp\esp32-slm-completer\firmware -B D:\esp\build-slm "-DPYTHON=C:\Espressif\tools\python\v6.1\venv\Scripts\python.exe" -DESP_PLATFORM=1   # 처음 한 번
ninja -C D:\esp\build-slm
cd D:\esp\build-slm; python -m esptool --chip esp32s3 -p COM9 -b 921600 write-flash "@flash_args" 0x190000 D:\esp\esp32-slm-completer\firmware\model.bin   # 모델이 그대로면 0x190000 부분 생략
```

`sdkconfig.defaults`를 고치면 `firmware/sdkconfig`를 지우고 cmake를 다시 돌린다 (남아 있으면 반영되지 않는다).

v6부터 cJSON은 ESP-IDF 본체에 없어서 `firmware/main/idf_component.yml`의 `espressif/cjson`으로 받는다.

## 필수 조건 활용

1. **ESP32에 모델**: 보드가 자동완성 서버. 응답에 생성 시간·토큰 수·최소 남은 힙을 싣는다.
2. **로그를 DB에**: 버전(모델·펌웨어·브리지) 조합별 실사용 성적표. 수락률(화면 표시 대비), 빈 완성 비율과 멈춘 이유, 지연, 입력 버림 비율, 문법 오류율, 최소 힙. 행은 추가만 한다. 키는 `bridge/.env`(git 제외). 버전은 손으로 정하지 않는다: 모델은 export.py가 내용 해시로, 펌웨어는 ESP-IDF가 git describe로, 브리지는 실행 때 git describe로 정한다.
3. **CI**: 빌드·테스트에 더해, `model-eval` job이 `firmware/model.bin`을 numpy로 돌려 예시 다음 줄 정확도 0.95 미만 또는 MBPP heldout 문법 통과율 0.85 미만이면 실패시킨다.
