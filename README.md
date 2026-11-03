# esp32-slm-completer

ESP32-S3에 올린 소형 언어 모델(SLM)로 VS Code에서 Python 코드를 자동완성한다.
보드를 USB로 연결하면 VS Code 확장이 보드를 찾아 커서 앞 코드를 보내고, 보드가 만든 완성을 회색 인라인 제안으로 보여 준다.
제안과 수락 기록은 Supabase에 쌓여 모델 버전별 실사용 성적표가 된다.

## 현재 상태 (2026-11-03)

보드의 모델 자리는 아직 고정 문자열(`  # slm stub`, `model_version: "stub"`)을 돌려준다. SLM 학습과 추론은 `dev` 브랜치에서 진행한다.

| 구간 | 상태 | 확인 방법 |
|---|---|---|
| 펌웨어 빌드·굽기 | 동작 | ESP-IDF v6.1 로컬 빌드, 보드에 굽고 응답 확인 |
| 브리지 ↔ 보드 | 동작 | 요청 2건·수락 1건 왕복, 지연 2~3 ms, 최소 남은 힙 약 388 KB |
| 브리지 → Supabase | 동작 | 3행 전송, 재전송 시 중복 없음, anon 읽기 차단 확인 |
| 확장 ↔ 브리지 | 동작 (가짜 브리지) | 실제 VS Code 통합 테스트 1건 통과 |
| 설치본 확장 ↔ 보드 | 확인 전 | `.vsix` 설치까지 완료, VS Code 화면에서 제안이 뜨는지는 아직 확인하지 않음 |
| CI | 통과 (확장 통합 테스트 추가 전 기준) | 확장 패키징·통합 테스트 job은 아직 GitHub에서 실행하지 않음 |
| SLM | 없음 | |

## 구성

```
VS Code 확장 (TypeScript)
  │  Python 파일에서 커서 앞 코드(최대 1024자) → 인라인 제안
  │  stdin/stdout, JSON 한 줄씩
  ▼
브리지 (Python)
  │  Espressif USB(VID 303A) 포트 자동 탐지, 요청 중계, 왕복 지연 측정
  │  로그를 logs/events.jsonl에 남기고 10초마다 Supabase로 전송
  ▼  USB 시리얼
ESP32-S3 펌웨어 (ESP-IDF v6.1)
     완성 생성, 생성 시간·토큰 수·최소 남은 힙을 응답에 포함
```

| 폴더 | 내용 |
|---|---|
| `firmware/` | ESP-IDF 펌웨어 (`main/main.c`, cJSON 의존성 `main/idf_component.yml`) |
| `bridge/` | 확장과 보드 사이 중계, 로그 수집·전송 (`bridge.py`, 테스트 `test_bridge.py`, 설정 예시 `.env.example`) |
| `extension/` | VS Code 확장 (`src/extension.ts`), 통합 테스트 `src/test/`, 가짜 브리지 `test-fixture/bridge.py` |
| `db/schema.sql` | Supabase `events` 테이블, 권한, 성적표 쿼리 |
| `.github/workflows/ci.yml` | CI |
| `.mcp.json` | Claude Code용 Supabase MCP 서버 등록 (키 없음) |
| `CLAUDE.md` | AI 작업용 프로젝트 안내 |

## 통신 형식

모든 메시지는 JSON 한 줄이다.

| 방향 | 메시지 |
|---|---|
| 확장 → 브리지 → 보드 | `{"id": 1, "prefix": "for i in range("}` |
| 보드 → 브리지 → 확장 | `{"id": 1, "completion": "10):", "model_version": "stub", "gen_ms": 0, "tokens": 0, "heap_min": 388612}` |
| 확장 → 브리지 (제안을 Tab으로 수락) | `{"type": "accept", "id": 1}` |
| 브리지 → 확장 (시작) | `{"ready": true, "port": "COM9", "session": "41ba8432"}` |

보드는 부팅 로그 등 JSON이 아닌 줄도 내보내며, 브리지는 이를 버린다.

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
python -m esptool --chip esp32s3 -p COM9 -b 460800 write-flash "@flash_args"
```

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

브리지가 제안(`suggest`)과 수락(`accept`)을 각각 한 행으로 `bridge/logs/events.jsonl`에 추가하고, 10초마다와 종료할 때 아직 보내지 않은 줄을 Supabase `events` 테이블로 보낸다. 보낸 위치는 `bridge/logs/events.offset`에 기록하므로 전송에 실패한 줄은 다음 차례에 다시 보낸다.

| 컬럼 | 내용 |
|---|---|
| `session`, `req`, `event` | 기본 키. 브리지 실행마다 새 `session` |
| `ts` | 브리지가 기록한 시각 |
| `model_version` | 보드 펌웨어의 모델 버전 |
| `prefix_tail`, `completion` | 커서 앞 코드의 끝 200자, 완성 결과 |
| `latency_ms` | 브리지 기준 왕복 시간 (USB 포함) |
| `gen_ms`, `tokens`, `heap_min` | 보드에서 잰 생성 시간, 생성 토큰 수, 최소 남은 힙 |

공개 키(anon)는 `events`에 insert만 할 수 있고 읽을 수 없다. 이 때문에 `ON CONFLICT`(중복 무시)를 쓸 수 없어서, 브리지는 일반 insert를 하고 409(중복)를 받으면 한 행씩 다시 넣으며 이미 있는 행을 건너뛴다.

모델 버전별 제안 수, 수락률, 평균 지연, 최소 힙은 `db/schema.sql` 끝의 쿼리로 조회한다.

## CI

GitHub Actions에서 push와 pull request마다 실행한다.

| job | 내용 |
|---|---|
| `bridge` | `uv run pytest` (5건): 포트 탐지, 로그 전송·재전송, 쓰는 중인 줄 건너뛰기, 409 중복 처리 |
| `extension` | `.vsix` 패키징, 실제 VS Code(xvfb)에서 Python 파일 → 제안 → 수락 → 브리지 accept 통합 테스트 |
| `firmware` | `espressif/esp-idf-ci-action`으로 ESP-IDF v6.1, esp32s3 빌드 |
