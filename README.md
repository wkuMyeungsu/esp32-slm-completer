# esp32-slm-completer

ESP32-S3에 올린 소형 언어 모델(SLM)로 VS Code에서 Python 코드를 자동완성한다.
보드를 USB로 연결하면 VS Code 확장이 보드를 찾아 커서 앞 코드를 보내고, 보드가 만든 완성을 회색 인라인 제안으로 보여 준다.
제안과 수락 기록은 Supabase에 쌓여 모델 버전별 실사용 성적표가 된다.

## 현재 상태 (2026-11-03)

보드가 Pythia-14M을 어휘 1,012개로 줄여 MBPP(초급 Python 문제)와 직접 만든 예시 8개로 학습한 모델(`pythia-14m-general-k768-s300-40b80eb5`, float32 5.8MB)로 완성을 만든다. 학습 과정과 숫자는 `train/LOG.md`에 있다.

| 평가 (numpy, 보드와 같은 계산) | 예시 8개 (51줄) 정확 일치 | MBPP heldout 100문제 (700줄) 정확 일치 | heldout 문법 통과 |
|---|---|---|---|
| 예시 8개만 외운 모델 (`pythia-14m-ft-0bbd3425`) | 1.000 | 0.000 | 0.629 (입력 토큰 47.9% 버림) |
| 지금 모델 | 1.000 | 0.117 | 0.912 (버림 0, 7.6% 쪼갬) |

| 구간 | 상태 | 확인 방법 |
|---|---|---|
| 펌웨어 빌드·굽기 | 동작 | ESP-IDF v6.1 로컬 빌드, 보드에 굽고 응답 확인 |
| 브리지 ↔ 보드 | 동작 | 앞부분 → 토큰화(쪼개기)·힐링 → 보드 추론 → 한 줄 완성 |
| 브리지 → Supabase | 동작 | `sessions`·`suggestions`·`feedback` 세 테이블, 재전송 시 중복 없음, anon 읽기 차단, security advisor 경고 없음 |
| 확장 ↔ 브리지 | 동작 (가짜 브리지) | 실제 VS Code 통합 테스트 1건 통과 (shown → accept) |
| 설치본 확장 ↔ 보드 | 동작 | VS Code에서 제안·수락, DB 기록 확인 (예시만 외운 모델로) |
| SLM (보드) | 동작 | 예시 51줄 정확도 1.000, 평균 1,117 ms, 최소 남은 힙 1,188,740 B (PSRAM 포함), `train/board_check.py` |

## 구성

```
VS Code 확장 (TypeScript)
  │  Python 파일에서 커서 앞 코드(최대 1024자) → 인라인 제안
  │  stdin/stdout, JSON 한 줄씩
  ▼
브리지 (Python)
  │  Espressif USB(VID 303A) 포트 자동 탐지, 요청 중계, 왕복 지연 측정
  │  앞부분을 Pythia 토크나이저로 잘라 보드의 줄인 어휘 id로 바꾸고, 토큰 힐링 문자열을 붙인다
  │  로그를 logs/events.jsonl에 남기고 10초마다 Supabase로 전송
  ▼  USB 시리얼
ESP32-S3 펌웨어 (ESP-IDF v6.1)
     플래시 model 파티션의 가중치를 PSRAM에 올려 GPT-NeoX 추론 (KV 캐시, 이전 요청과 겹치는 앞부분 재사용)
     완성 생성, 생성 시간·토큰 수·최소 남은 힙을 응답에 포함
```

| 폴더 | 내용 |
|---|---|
| `firmware/` | ESP-IDF 펌웨어 (`main/main.c` 통신, `main/model.c` 추론, cJSON 의존성 `main/idf_component.yml`), 파티션 표 `partitions.csv`, 가중치 `model.bin` |
| `bridge/` | 확장과 보드 사이 중계, 토큰화, 로그 수집·전송 (`bridge.py`, 줄인 어휘 표 `vocab.json`, 테스트 `test_bridge.py`, 설정 예시 `.env.example`) |
| `train/` | 학습 (`finetune.py`), 보드용 내보내기 (`export.py`), numpy 추론 (`npmodel.py`), 평가 (`eval_bin.py`, `board_check.py`), 학습 예시 `data/`, 기록 `LOG.md` |
| `extension/` | VS Code 확장 (`src/extension.ts`), 통합 테스트 `src/test/`, 가짜 브리지 `test-fixture/bridge.py` |
| `db/schema.sql` | Supabase `events` 테이블, 권한, 성적표 쿼리 |
| `.github/workflows/ci.yml` | CI |
| `.mcp.json` | Claude Code용 Supabase MCP 서버 등록 (키 없음) |
| `CLAUDE.md` | AI 작업용 프로젝트 안내 |

## 통신 형식

모든 메시지는 JSON 한 줄이다.

| 방향 | 메시지 |
|---|---|
| 확장 → 브리지 | `{"id": 1, "prefix": "for i in range("}` |
| 브리지 → 보드 | `{"id": 1, "tokens": [...], "heal": "("}` (줄인 어휘 id, 뗀 마지막 토큰의 문자열) |
| 보드 → 브리지 → 확장 | `{"id": 1, "completion": "1, 10):", "stop": "newline", "model_version": "pythia-14m-ft-0bbd3425", "gen_ms": 619, "tokens": 6, "heap_min": 2113284}` |
| 보드 → 브리지 (오류) | `{"id": 1, "error": "bad request: no tokens"}` (id를 알 수 없으면 id 없이) |
| 브리지 → 보드 (시작 때 버전 요청) | `{"id": 0, "info": true}` |
| 보드 → 브리지 | `{"id": 0, "fw_version": "589d8d3-dirty", "fw_build": "a1b2c3d4", "model_version": "pythia-14m-ft-0bbd3425", "vocab": 152}` |
| 확장 → 브리지 (제안을 VS Code에 넘김) | `{"type": "shown", "id": 1}` |
| 확장 → 브리지 (제안을 Tab으로 수락) | `{"type": "accept", "id": 1}` |
| 브리지 → 확장 (시작) | `{"ready": true, "port": "COM9", "session": "41ba8432"}` |

보드는 부팅 로그 등 JSON이 아닌 줄도 내보내며, 브리지는 이를 진단 로그에 DEBUG로 남기고 통신에는 쓰지 않는다.

브리지는 시작할 때 보드의 `model_version`과 `bridge/vocab.json`의 `model_version`을 비교한다. 다르면 토큰 id가 엉뚱하게 매핑되므로 오류를 내고 끝난다.

## 하드웨어

| 항목 | 값 |
|---|---|
| 칩 | ESP32-S3 (QFN56) rev v0.2, 240MHz 듀얼코어 |
| PSRAM | 8MB 내장 |
| 플래시 | 8MB |
| 연결 | 내장 USB-Serial/JTAG |

## 실행

### 1. 펌웨어 빌드·굽기

ESP-IDF v6.1이 필요하다. cJSON은 v6부터 ESP-IDF 본체에 없어서 `firmware/main/idf_component.yml`의 `espressif/cjson`으로 받는다.

경로에 한글이나 공백이 없으면 일반적인 방법으로 된다.

```
cd firmware
idf.py set-target esp32s3
idf.py -p COM9 flash monitor
```

경로에 한글이 있으면 `idf.py`의 cmake 단계가 비정상 종료한다(0xC0000409). `idf.py`는 junction을 실제 경로로 바꿔 쓰므로, 영문 경로 junction에서 cmake와 ninja를 직접 부른다.

```powershell
New-Item -ItemType Junction -Path D:\esp\esp32-slm-completer -Target <저장소 경로>
. C:\Espressif\tools\Microsoft.v6.1.PowerShell_profile.ps1
cmake -G Ninja -S D:\esp\esp32-slm-completer\firmware -B D:\esp\build-slm "-DPYTHON=C:\Espressif\tools\python\v6.1\venv\Scripts\python.exe" -DESP_PLATFORM=1
ninja -C D:\esp\build-slm
cd D:\esp\build-slm
python -m esptool --chip esp32s3 -p COM9 -b 921600 write-flash "@flash_args" 0x190000 D:\esp\esp32-slm-completer\firmware\model.bin
```

`model.bin`은 앱과 따로 `model` 파티션(0x190000, `partitions.csv`)에 굽는다. 모델이 바뀌지 않았으면 `0x190000 ...` 부분을 빼고 앱만 구워도 된다(4.9MB 굽기에 약 33초).

`sdkconfig.defaults`를 고쳤다면 `firmware/sdkconfig`를 지우고 cmake를 다시 돌린다. sdkconfig가 남아 있으면 새 기본값이 반영되지 않는다.

### 모델 학습과 내보내기

```
cd train
uv run train_general.py --k 768 --steps 300 --eval-every 25 --name pythia-14m-general-k768-s300   # → ckpt/<이름>
uv run export.py ckpt/pythia-14m-general-k768-s300   # → ../firmware/model.bin, ../bridge/vocab.json, torch 대비 확인
cd ..
uv run --no-project --with numpy --with tokenizers python train/eval_bin.py   # examples·heldout 평가 (CI와 같음)
cd train
uv run board_check.py                   # 보드에 구운 뒤 예시 51줄로 정확도·속도 측정
```

- 데이터 (`train/corpus.py`): 직접 만든 예시 8개(`train/data/`)와 MBPP (Austin et al., 2021, "Program Synthesis with Large Language Models", CC BY 4.0, `train/data_ext/mbpp.jsonl`, google-research/google-research 저장소). MBPP 문제 설명을 첫 줄 주석으로 붙이고 탭을 공백 4칸으로 바꾼다. task_id 411~510(100문제)은 평가 전용이라 학습에 쓰지 않는다.
- 어휘: 학습 코드에서 자주 나오는 토큰 768개 + 1바이트 토큰 243개 + EOS = 1,012개. 학습 전에 모델의 입력·출력 임베딩을 이 행만 남긴다. 어휘에 없는 토큰은 버리지 않고 어휘 안의 더 작은 토큰(최악은 1바이트)으로 쪼갠다. 규칙은 `bridge/vocab_map.py` 하나를 학습·평가·브리지가 함께 쓴다.
- 문법 판정은 `bridge/syntax.py` 하나를 브리지(실사용 기록)와 평가가 함께 쓴다.
- 학습·평가 로그는 `train/logs/`에, 시행착오와 숫자는 `train/LOG.md`에 남긴다.
- `finetune.py`는 예시 8개만 외우는 처음 방식이다 (전체 어휘, `export.py`가 예시에 나온 토큰만 남기고 나머지는 버린다).

버전은 손으로 정하지 않는다. `export.py`가 `<체크포인트 이름>-<내용 sha256 앞 8자리>`를 `model.bin` 끝과 `bridge/vocab.json`에 함께 쓰고, 펌웨어 버전은 빌드 때 ESP-IDF가 `git describe`로 넣는다. 모델을 다시 내보내면 `model.bin`을 다시 굽는다(버전이 달라져 브리지가 어긋남을 알린다).

### 2. Supabase

1. 프로젝트 설정: Data API 켜기, 새 테이블 자동 공개 끄기, 자동 RLS 켜기.
2. SQL Editor에서 `db/schema.sql`을 실행한다.
3. 자동 RLS를 켜면 생기는 `public.rls_auto_enable()` 함수는 기본 권한(PUBLIC)으로 anon·authenticated가 API로 실행할 수 있다(Security Advisor 경고). 실행 권한을 회수한다. 회수한 뒤에도 새 테이블에 RLS가 자동으로 켜지는 것을 확인했다.
   ```sql
   revoke execute on function public.rls_auto_enable() from public, anon, authenticated;
   ```
4. `bridge/.env.example`을 `bridge/.env`로 복사하고 Project URL과 publishable 키를 넣는다. `.env`는 git에 올라가지 않는다.

`.env`가 없으면 브리지는 로그를 파일에만 남긴다.

### 3. 브리지

[uv](https://docs.astral.sh/uv/)가 필요하다. 확장이 `uv run bridge.py`로 직접 실행하므로 따로 띄울 필요는 없다. 보드와의 연결만 확인하려면 다음과 같이 한다.

```powershell
cd bridge
'{"id":1,"prefix":"print("}' | uv run bridge.py
```

시리얼 포트는 한 프로그램만 열 수 있다. `idf.py monitor` 등 다른 프로그램이 포트를 쓰고 있으면 브리지가 연결하지 못한다.

### 4. VS Code 확장

설치본으로 쓰기:

```
cd extension
npm install
npm run package
code --install-extension esp32-slm-completer-0.0.1.vsix --force
```

설치본은 확장 폴더가 `~/.vscode/extensions`로 옮겨지므로, VS Code 설정 `esp32Slm.bridgeDir`에 저장소 `bridge` 폴더의 절대 경로를 넣는다. 개발 모드로 쓰려면 `extension` 폴더를 VS Code로 열고 F5(Run Extension)를 누른다. 이때는 설정이 비어 있어도 `../bridge`를 쓴다.

`.py` 파일에 코드를 입력하면 인라인 제안이 뜨고, Tab으로 수락한다. 확장의 출력은 출력 패널의 `ESP32 SLM` 채널에서 볼 수 있다.

확장은 셸로 `uv run bridge.py`를 실행한다. Windows에서 uv가 pyenv-win shim(`uv.bat`)이면 셸 없이 실행할 수 없기 때문이다. 확장이 꺼질 때는 브리지의 stdin을 닫아, 브리지가 남은 로그를 보낸 뒤 스스로 끝나게 한다.

브리지는 확장이 켜질 때 한 번 보드를 찾는다. 보드를 나중에 연결했다면 창을 다시 띄운다(`Developer: Reload Window`).

통합 테스트는 실제 VS Code를 띄워, 보드 대신 `extension/test-fixture/bridge.py`(가짜 브리지)를 붙여 제안과 수락을 확인한다.

```
npm test
```

테스트용 VS Code를 내려받을 수 없는 환경이면 `VSCODE_EXECUTABLE_PATH`에 설치된 `Code.exe` 경로를 넣는다.

## 로그와 DB

브리지가 기록을 `bridge/logs/rows.jsonl`에 바로 한 줄씩 남기고, 10초마다와 종료할 때 아직 보내지 않은 줄을 Supabase의 각 테이블로 보낸다(세션 → 제안 → 반응 순서). 보낸 위치는 `bridge/logs/rows.offset`에 기록하므로 전송에 실패한 줄은 다음 차례에 다시 보낸다. 칸 설명은 `db/schema.sql`에 있다.

| 테이블 | 한 행 = | 주요 칸 |
|---|---|---|
| `sessions` | 브리지 실행 1회 | `bridge_version`, `fw_version`, `fw_build`, `model_version`, `vocab_size`, `port` |
| `suggestions` | 제안 1건 | `completion`, `cached`, `latency_ms`, `gen_ms`, `prompt_tokens`, `dropped_tokens`(어휘에 없어 버린 입력), `gen_tokens`, `stop_reason`(멈춘 이유), `heap_min`, `syntax`·`syntax_error`(문법 검사) |
| `feedback` | 사용자 반응 1건 | `kind` = `shown`(VS Code에 넘김) 또는 `accept`(Tab으로 수락) |

문법 검사는 브리지가 표준 라이브러리 `codeop`(대화형 Python이 `...`을 띄울지 정하는 모듈)로 '앞부분 + 완성'을 검사해 `ok`, `incomplete`(블록·괄호가 이어질 차례), `error`, `prefix_error`(완성 전 줄들이 이미 오류)로 나눈다. 앞부분이 1024자로 잘린 경우는 판단하지 않는다.

조회는 Supabase 대시보드 SQL Editor에서 한다.

| 뷰 | 내용 |
|---|---|
| `scorecard` | 모델·펌웨어·브리지 버전 조합별 제안 수, 화면 표시 대비 수락률, 빈 완성 비율, 평균 지연·생성 시간, 입력 버림 비율, 문법 오류율(전체·수락한 제안) |
| `stop_reasons` | 버전별 멈춘 이유 분포와 그중 빈 완성 수 |

공개 키(anon)는 세 테이블에 insert만 할 수 있고 읽을 수 없다. 이 때문에 `ON CONFLICT`(중복 무시)를 쓸 수 없어서, 브리지는 일반 insert를 하고 409(중복)를 받으면 한 행씩 다시 넣으며 이미 있는 행을 건너뛴다. 13:50 이전 기록이 담긴 예전 `events` 테이블은 지우지 않고 남겨 두었고, 데이터는 세 테이블로 옮겼다.

진단 로그는 따로 남긴다. 형식은 `시각 레벨 로거이름 파일:줄 메시지`다.

| 파일 | 내용 |
|---|---|
| `bridge/logs/bridge.log` | 브리지 진단 로그. DEBUG 이상, 1MB × 4개 로테이션. 업로드 실패·포트 열기 실패·시리얼 오류(Traceback 포함), 보드 시간 초과·오류 응답, 요청별 토큰 수·캐시 사용, 보드의 부팅·ESP_LOG 출력. INFO 이상은 stderr로도 내보내 VS Code 출력 채널 `ESP32 SLM`에 보인다 |
| `train/logs/<스크립트>-<시각>.log` | 학습·평가 스크립트 로그. DEBUG 이상 (finetune은 매 스텝 loss). 콘솔에는 INFO 이상 |

`events.jsonl`은 DB에 올릴 데이터라 로테이션하지 않고, `bridge.log`만 로테이션한다. 실시간으로 보려면 `Get-Content bridge\logs\bridge.log -Wait -Tail 20`.

## CI

GitHub Actions에서 push와 pull request마다 실행한다.

| job | 내용 |
|---|---|
| `bridge` | `uv run pytest` (6건): 포트 탐지, 로그 전송·재전송, 쓰는 중인 줄 건너뛰기, 409 중복 처리, 토큰 힐링 |
| `train-data` | 학습 예시가 실행되고 기대 출력을 내는지 |
| `model-eval` | `firmware/model.bin`을 numpy로 돌려 예시 다음 줄 정확도가 0.95 미만이거나 MBPP heldout 문법 통과율이 0.85 미만이면 실패 (torch 없이 numpy·tokenizers) |
| `extension` | `.vsix` 패키징, 실제 VS Code(xvfb)에서 Python 파일 → 제안 → 수락 → 브리지 accept 통합 테스트 |
| `firmware` | `espressif/esp-idf-ci-action`으로 ESP-IDF v6.1, esp32s3 빌드 |
