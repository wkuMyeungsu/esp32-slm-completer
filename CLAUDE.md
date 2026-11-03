# esp32-slm-completer

## 공통 컨텍스트 메모리

`.agents/memory.txt`에 이 프로젝트의 공통 컨텍스트 메모리 로그가 있다. 작업을 시작하기 전에 읽고, 진행 상황이 바뀌면 한 줄씩 덧붙인다 (`YYYY-MM-DD HH:MM 내용`).

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
| `firmware/` | ESP-IDF 펌웨어. USB 시리얼로 JSON 한 줄 요청 `{"id","prefix"}`을 받아 `{"id","completion","model_version","gen_ms","tokens","heap_min"}`으로 답한다 |
| `bridge/` | Python 브리지. Espressif USB(VID 303A) 포트를 찾아 확장과 보드 사이를 중계하고, 로그를 `logs/events.jsonl`에 남긴 뒤 10초마다 Supabase로 보낸다 |
| `extension/` | VS Code 확장. Python 파일에서 커서 앞 코드를 보내고 인라인 제안으로 보여 준다. 제안을 수락하면 브리지에 `{"type":"accept","id"}`를 보낸다 |
| `db/schema.sql` | Supabase `events` 테이블, anon은 insert만 허용, 모델 버전별 성적표 SQL |
| `.github/workflows/ci.yml` | CI: 브리지 pytest, 확장 tsc, 펌웨어 ESP-IDF 빌드 |

## 펌웨어 빌드·굽기 (로컬, ESP-IDF v6.1)

저장소 경로에 한글이 있어 `idf.py build`는 cmake가 비정상 종료한다(0xC0000409). `idf.py`는 junction을 따라가므로 소용없고, 영문 junction `D:\esp\esp32-slm-completer`로 cmake·ninja를 직접 부른다.

```powershell
. C:\Espressif\tools\Microsoft.v6.1.PowerShell_profile.ps1
cmake -G Ninja -S D:\esp\esp32-slm-completer\firmware -B D:\esp\build-slm "-DPYTHON=C:\Espressif\tools\python\v6.1\venv\Scripts\python.exe" -DESP_PLATFORM=1   # 처음 한 번
ninja -C D:\esp\build-slm
cd D:\esp\build-slm; python -m esptool --chip esp32s3 -p COM9 -b 460800 write-flash "@flash_args"
```

v6부터 cJSON은 ESP-IDF 본체에 없어서 `firmware/main/idf_component.yml`의 `espressif/cjson`으로 받는다.

## 필수 조건 활용

1. **ESP32에 모델**: 보드가 자동완성 서버. 응답에 생성 시간·토큰 수·최소 남은 힙을 싣는다.
2. **로그를 DB에**: 모델 버전별 실사용 성적표(제안 수, 수락률, 지연, 최소 힙). 제안(`suggest`)과 수락(`accept`)을 각각 한 행으로 추가만 한다. 키는 `bridge/.env`(git 제외).
3. **CI**: 빌드·테스트에 더해, 평가용 예제의 앞부분을 완성시켜 Python 문법 통과율·원본 일치율이 기준 미만이면 실패시킨다 (추가 예정).
