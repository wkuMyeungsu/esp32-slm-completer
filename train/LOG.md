# 학습 기록

시행착오를 시간 순서로 남긴다. 항목마다 한 일, 설정, 결과(숫자), 원인, 조치를 적는다. 새 시도는 맨 아래에 이어 쓴다.

환경: RTX 3060 12GB, torch 2.11.0+cu128, transformers 5.22.1 (`train/pyproject.toml`)

## 2026-11-03 11:00 모델 후보 조사

ESP32-S3(플래시 8MB, PSRAM 8MB, 내부 SRAM 남은 힙 약 388KB)에 넣을 수 있고 코드를 학습한 모델을 Hugging Face에서 찾았다. 4천만 파라미터 이하 생성 모델 중 테스트용 랜덤 가중치 모델을 빼고, 모델 카드에서 학습 데이터를 확인했다.

| 모델 | 파라미터 | 구성 | 학습 데이터 | 표준 transformers |
|---|---|---|---|---|
| EleutherAI/pythia-14m | 14.1M | dim 128, 6층, 어휘 50,304 (입·출력 분리) | The Pile (GitHub 포함) | 예 (gpt_neox) |
| EleutherAI/pythia-31m | 30.5M | dim 256, 6층, 어휘 50,304 | The Pile (GitHub 포함) | 예 |
| SmallDoge/Doge-20M | 13.1M | dim 256, 8층, 어휘 32,768 | smollm-corpus (python-edu 포함) | 아니오 (trust_remote_code) |
| MinimaLabs/min-spark-1.5 | 9.4M | 어휘 4,096, 블록 공유·반복 | smollm-corpus 등 | 아니오 (trust_remote_code) |

코드를 학습하지 않아 뺀 모델: pico-decoder-tiny(Dolma 영어), Tiny-LLM·gpt2-mini(웹 텍스트), nanochat-40M(대화), TinyStories 계열(동화). SmolLM2-135M은 본체만 약 1억 파라미터라 뺐다.

결정: Pythia-14M 1순위, Pythia-31M 다음 단계. 표준 구조라 원격 코드 실행이 없고 C 추론을 직접 짤 수 있다. 14M의 92%가 임베딩이라 어휘를 줄이면 크기가 크게 준다.

## 2026-11-03 11:45 기준선: 원본 모델의 Python 이어쓰기

`uv run baseline.py EleutherAI/pythia-14m EleutherAI/pythia-31m` (탐욕적 생성, 24토큰)

| 입력 | Pythia-14M | Pythia-31M |
|---|---|---|
| `# print hello world\nprint(` | `0, "http://www.com/news/news/...` | `hello world)\nprint(hello world)\n...` |
| `for i in range(` | `i, $i,1,i,1,i,1,...` | `0,0)\n    for i in range(0,0):\n        if i == 0:` |
| `def add(a, b):\n    return` | ` b.b.b.b.b.b...` | ` add(a, b)\n\ndef add(a, b):\n    return add(a, b)` |
| `...for n in numbers:\n    total` | ` = 0\n    total = 0\n...` | ` = number\n    total = number\n...` |
| `...print(a +` | ` b + b + b + b)\n\nA:\n\nThe problem is...` | ` 1)\nprint(b + 1)\nprint(b + 1)...` |

- 14M: 괄호를 닫지 못하고 영어 문장·URL로 샌다. 같은 토큰을 반복한다.
- 31M: 괄호를 닫고 `for ... :` 다음 줄을 들여쓰고 `def`·`return` 짝을 맞춘다. 내용은 틀리다(`range(0,0)`, 자기 자신 호출, 따옴표 없는 문자열).
- 둘 다 같은 줄을 반복한다.

## 2026-11-03 12:05 VS Code에서 원본 Pythia-31M 사용

`serve_pc.py`(PC GPU에서 모델 실행, 확장과 같은 JSON 형식)로 VS Code 확장에 연결했다. 응답 0.2~0.5초.

- 처음에 추천이 뜨지 않았다. 원인은 VS Code가 폴더를 제한 모드(Restricted Mode)로 열어 우리 확장을 비롯해 신뢰 선언이 없는 확장을 모두 꺼 둔 것. 확장 호스트 로그에서 Pylance·Jupyter만 활성화되고 나머지는 활성화되지 않은 것으로 확인했다.
- 추천 예: `print("print" "print" "print" ...` 같은 반복, 맥락과 무관한 `finally:`.

## 2026-11-03 12:10 학습 데이터: 주제별 대표 예시 8개

변형을 많이 만드는 대신 주제마다 완성된 예시 하나를 외우게 하기로 했다. 외워서 되살리는 방식이라 정확하고 평가가 쉽고, 어휘가 수백 개로 줄어 보드에 넣기 좋다. 예시와 다르게 쓴 코드에는 약하다.

`train/data/`: hello, variables, grade, times_table, while_sum, list_stats, factorial, calculator. 첫 줄은 서로 다른 영어 주석, 실행 결과가 항상 같게(난수·파일 쓰기 제외), 10~20줄, 주석·문자열은 영어(Pythia 토크나이저가 한글을 바이트 단위로 쪼갬). 예시 중 가장 긴 것이 144토큰. `test_data.py`가 실행 결과를 확인한다.

## 2026-11-03 12:10 시행착오: loss가 nan

`finetune.py`를 2스텝 돌리자 step 1 loss 3.2333 다음부터 nan.

- 원인: transformers 5는 체크포인트에 저장된 dtype으로 불러온다. Pythia는 float16으로 저장돼 있어 float16 가중치로 AdamW 학습을 하다가 값이 넘쳤다. `next(model.parameters()).dtype`이 `torch.float16`인 것으로 확인했다.
- 조치: `from_pretrained(..., dtype=torch.float32)`로 불러온다.
- 확인: 40스텝, loss 3.2299 → 0.0440, 다음 줄 정확도 0.961.

## 2026-11-03 12:13 1차 학습: Pythia-14M, 300스텝

`uv run finetune.py --model EleutherAI/pythia-14m --steps 300 --lr 3e-4` (예시 8개 한 배치, AdamW, seed 0) → `ckpt/pythia-14m-ft`

| step | 1 | 20 | 40 | 60 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|
| loss | 3.2299 | 0.1136 | 0.0440 | 0.0372 | 0.0360 | 0.0355 | 0.0354 |

다음 줄 정확도 0.961 (51줄 중 49줄). 학습한 예시로 평가했으므로 외운 정도를 뜻한다.

- loss가 0.035 근처에서 멈춘다. 첫 줄 주석만으로는 8개 중 어느 예시인지 정할 수 없어 그 불확실성이 남는다.
- 틀린 2줄은 둘 다 빈 줄 두 개 바로 다음 줄이다: `print(factorial(5))`, `a = int(input("First number: "))`. 생성 결과는 빈 문자열.

## 2026-11-03 12:16 시행착오: 앞부분이 공백으로 끝나면 토큰이 학습 때와 다르게 끊긴다

1차 학습에서 틀린 2줄의 원인. 같은 문자열이 뒤에 글자가 오는지에 따라 다르게 토큰화된다.

```
학습 때: ')'  '\n\n'  '\n'  'print'  '('  'factor'  'ial' ...
평가 때: ')'  '\n\n\n'                        (앞부분이 여기서 끝남)
```

Pythia(GPT-NeoX) 토크나이저는 공백 덩어리 뒤에 글자가 오면 마지막 공백 하나를 떼어 둔다. 앞부분이 공백으로 끝나면 학습 때 나오지 않은 토큰(`\n\n\n`)이 마지막에 오고, 모델은 다음 줄 대신 빈 문자열을 낸다.

VS Code에서는 Enter를 치면 자동 들여쓰기가 들어가 커서 앞 코드가 `":\n    "`처럼 공백으로 끝나는 경우가 많아, 평가보다 실제 사용에서 더 자주 생긴다.

조치 후보(토큰 힐링): 앞부분의 마지막 토큰이 공백뿐이면 떼고 생성한 뒤, 생성 결과가 그 공백으로 시작하는지 확인하고 공백을 잘라 추천으로 준다. 학습은 그대로 두고 추천하는 쪽(`eval_lines.py`, `serve_pc.py`, 펌웨어)만 고친다.

## 2026-11-03 12:22 보드용으로 줄이기: 어휘 50,304 → 152, numpy 순전파

`uv run export.py ckpt/pythia-14m-ft` → `ckpt/pythia-14m-ft/model.bin` (4,915,982 바이트), `vocab.json`

- 학습 예시 8개에 나온 토큰 151개 + EOS만 남기고 `embed_in`·`embed_out`의 해당 행만 잘랐다. 층 6개(약 119만 파라미터)는 그대로. 합계 약 123만 파라미터, float32.
- GPT-NeoX 순전파를 numpy로 한 토큰씩(KV 캐시) 다시 짰다. 회전 위치 임베딩은 헤드 차원 32 중 앞 8차원만, 어텐션과 MLP는 parallel residual, GELU는 erf. 펌웨어 C 코드가 이 구현을 옮긴다.
- torch(전체 어휘 로짓에서 남긴 열만)와 비교: 로짓 최대 오차 2.71e-04, 12토큰 모두 argmax 일치.
- 시행착오: `hf.state_dict()`에 `embed_out.weight` 키가 없어 KeyError. safetensors 파일을 직접 읽어 해결.

## 2026-11-03 12:24 토큰 힐링으로 다음 줄 정확도 0.961 → 1.000

numpy 모델, 학습 예시 51줄.

| 방식 | 다음 줄 정확도 |
|---|---|
| 힐링 없음 | 0.961 (49/51) |
| 힐링 있음 (줄인 어휘 id에서) | 0.961 (49/51) |
| 힐링 있음 (원래 id에서) | 1.000 (51/51) |

- 힐링: 앞부분의 마지막 토큰을 떼고, 생성 첫 토큰들을 그 문자열과 앞이 맞는 토큰으로 제한한 뒤, 생성 결과에서 그 문자열을 잘라 낸다.
- 처음엔 효과가 없었다. 원인은 문제의 끝 토큰 `'\n\n\n'`이 학습 데이터에 없어 줄인 어휘에도 없고, 어휘 변환 때 이미 버려져 힐링이 엉뚱한 `')'`를 떼고 있었던 것. 원래 토큰 id에서 먼저 힐링하고 나서 줄인 어휘로 바꾸게 고쳤다 (`export.py`의 `encode`).
- 줄인 어휘에 없는 나머지 토큰은 버린다. 학습 예시 밖의 코드에서는 앞부분이 빠질 수 있다.

## 2026-11-03 12:29 보드에서 추론: 다음 줄 정확도 1.000, 평균 761 ms

펌웨어 `firmware/main/model.c`가 numpy 구현을 C로 옮긴 것. 가중치는 플래시 `model` 파티션(0x190000, `partitions.csv`)에 `model.bin`을 따로 굽고, 부팅 때 PSRAM으로 읽는다. KV 캐시(최대 256위치, 1.5MB)도 PSRAM. 토큰화와 토큰 힐링 문자열은 PC 쪽이 만들어 `{"id", "tokens", "heal"}`로 보낸다.

굽기: `python -m esptool --chip esp32s3 -p COM9 -b 921600 write-flash "@flash_args" 0x190000 <ckpt>/model.bin` (model.bin 4.9MB에 33초)

확인: `uv run board_check.py` (학습 예시 51줄을 보드에 보내 numpy와 같은 기준으로 채점)

| 단계 | 첫 줄 (입력 3토큰 + 생성 9토큰) | 51줄 평균 | 정확도 |
|---|---|---|---|
| 처음 (PSRAM 40MHz, 데이터 캐시 32KB·라인 32B, 요청마다 처음부터 계산) | 1,759 ms (한 스텝 약 147 ms) | 재지 않음 (3줄 평균 2,080 ms) | 3/3 |
| PSRAM 80MHz, 데이터 캐시 64KB·라인 64B, 이전 요청과 겹치는 앞부분 KV 캐시 재사용 | 769 ms (한 스텝 약 64 ms) | 761 ms | 51/51 |

- 최소 남은 힙 2,113,284 바이트 (PSRAM 포함).
- 한 스텝마다 가중치 약 4.7MB를 PSRAM에서 읽는다. 계산(약 120만 곱셈·덧셈)보다 메모리 읽기가 병목이라 보고 메모리 설정부터 바꿨다. 첫 줄은 캐시 재사용이 없으므로 147 → 64 ms는 설정 효과.
- 캐시 재사용은 앞부분이 이전 요청과 같은 만큼 다시 계산하지 않는다. VS Code에서 타이핑하면 요청마다 앞부분이 거의 같아 효과가 크다. 입력이 130토큰이어도 1.2초.
- 시행착오: sdkconfig.defaults를 고쳐도 기존 `firmware/sdkconfig`가 있으면 반영되지 않는다. sdkconfig를 지우고 cmake를 다시 돌렸다.

## 2026-11-03 12:36 브리지 연결과 CI 평가 기준

- 토큰화와 토큰 힐링을 브리지로 옮겼다 (`tokenizers`로 `EleutherAI/pythia-14m` 토크나이저, 줄인 어휘 표 `bridge/vocab.json`). 브리지가 보드에 `{"id", "tokens", "heal"}`을 보낸다.
- 실제 브리지 → 보드 확인: `for i in range(` → `1, 10):` (619 ms). 처음엔 완성 끝에 `\n`이 붙어 나와(`"if n <= 1:\n"`) VS Code가 줄바꿈까지 넣게 되어 있었다. 보드에서 글자 뒤 첫 줄바꿈부터 자르게 고쳤다.
- 학습 예시와 다른 첫 줄 주석(`# Compute a factorial with a function`)에서는 외운 예시(`07_factorial.py`)의 줄을 그대로 낸다. `for` 안에서도 `if n <= 1:`을 제안했다. 외워서 되살리는 방식의 한계.
- 확장: 입력이 멈춘 뒤 300ms 기다렸다가 요청한다(디바운스). 응답 대기는 3초 → 15초 (캐시 없이 130토큰이면 보드에서 10초 가까이 걸린다). 브리지 시리얼 대기는 10초 → 30초.
- `model.bin`을 `firmware/model.bin`으로 저장소에 넣고, `export.py`가 쓴 파일을 `npmodel.py`로 다시 읽어 torch와 비교하게 했다 (로짓 최대 오차 2.71e-04, 파일 형식까지 확인). 다시 내보낸 파일은 보드에 구운 파일과 바이트 단위로 같았다.
- CI `model-eval`: `uv run --no-project --with numpy --with tokenizers python train/eval_bin.py --min 0.95`. 로컬 결과 1.000 (51/51). torch 없이 돈다.

## 2026-11-03 13:07 VS Code 실사용 테스트 (보드, 설치본 확장)

`train/playground/01_hello.py`에서 직접 입력. 세션 `8e8b0a31`, 제안 32건·수락 3건 (`bridge/logs/events.jsonl`, Supabase에도 같은 행).

잘 된 것:
- 예시대로 입력하면 한 글자씩 칠 때마다 남은 부분을 맞게 제안: `print(` → `"Hello, World!")`, `print("He` → `llo, World!")`, `    pri` → `nt(f"2 x {i} = {2 * i}")`.
- 지연은 대부분 0.2~0.7초. 들여쓰기 줄 전체를 만들 때(18~20토큰) 1.4~1.6초.

문제:

| 현상 | 예 (앞부분 끝 → 완성) |
|---|---|
| 같은 앞부분을 연달아 요청하고 매번 다시 생성 | `range(1, 10):\n    ` 3번 연속, 각 1.4초 |
| 줄 끝에서 완성이 줄바꿈 하나뿐 | `print("Hello, World!")` → `'\n'` (3건) |
| 예시와 다르게 친 내용을 무시하고 외운 줄을 냄 | 쉼표 없이 `"Hello World!"`라고 쳐도 다음 제안은 쉼표가 있는 `print("Hello, World!")` |
| 다른 예시의 줄이 섞임 | hello world 파일의 `for` 안에서 구구단 예시의 `print(f"2 x {i} = {2 * i}")` |
| 학습에 없는 입력에서 반복 | `    af` → `" = " = " = " = ...` (25토큰, 2초) |
| 빈 완성 | `\n\nfor` → `''` (2건) |

- 원인 (사실): 학습 데이터가 예시 8개뿐이고 어휘도 그 8개에 나온 152토큰뿐이다. 처음 보는 토큰은 브리지에서 버려지므로 모델은 사용자가 친 내용의 일부를 보지 못한다.

## 2026-11-03 13:17 같은 앞부분 재요청 캐시, 줄바꿈뿐인 완성 없애기

13:07 실사용 테스트에서 나온 문제 두 개를 고쳤다.

- 브리지: 직전 요청과 앞부분이 같으면 보드를 부르지 않고 직전 결과를 준다. 로그에는 `gen_ms 0`, `tokens 0`으로 남겨 보드가 일한 행과 구분한다. 보드 요청 부분을 `ask_board`로 빼고 테스트 1건 추가 (브리지 테스트 7건 통과).
- 펌웨어: 힐링 문자열을 잘라 낸 완성이 공백·줄바꿈뿐이면 빈 문자열을 돌려준다.
- 시행착오: VS Code 확장이 띄운 브리지가 COM9를 쥐고 있어 굽기가 `PermissionError(13)`으로 실패했다. VS Code 창을 닫아야 굽힌다.

확인 (사용자가 직접 굽고 실행, 세션 `cab39daa`):

| 요청 | 앞부분 | 완성 | gen_ms | tokens |
|---|---|---|---|---|
| 1 | `# hello world\nprint(` | `"Hello, World!")` | 762 | 7 |
| 2 | 1과 같음 | `"Hello, World!")` | 0 (캐시) | 0 |
| 3 | `# hello world\nprint("Hello, World!")` | `""` (고치기 전 `"\n"`) | 215 | 3 |

`uv run board_check.py`: 다음 줄 정확도 1.000 (51/51), 평균 생성 시간 760 ms (고치기 전 761 ms), 최소 남은 힙 2,113,284 바이트. 정확도와 속도는 그대로.

## 2026-11-03 13:26 로깅 정비 (Day11 Logging 수업 기준)

점검 결과 (고치기 전): 브리지·학습 스크립트 모두 `logging`을 쓰지 않았다. 브리지는 `print(..., file=sys.stderr)` 3곳, 업로드 실패는 `upload failed: {e}` 한 줄(Traceback 없음). 보드 응답 시간 초과와 보드의 JSON 아닌 출력(부팅·ESP_LOG·패닉)은 어디에도 남지 않았다. 학습 숫자는 터미널 출력을 손으로 옮겨 적었고 원본은 없다. 이 LOG의 시각 4개(12:22, 12:24, 12:36, 13:17)는 처음에 어림잡아 적었다가 `date`로 확인해 고쳤다.

바꾼 것:

| 대상 | 콘솔 | 파일 | 내용 |
|---|---|---|---|
| 학습·평가 (`train/log_setup.py`) | stdout, INFO | `train/logs/<스크립트>-<시각>.log`, DEBUG | finetune(매 스텝 loss는 파일, 20스텝마다 콘솔, loss가 nan·inf면 ERROR 후 중단), export, eval_bin(기준 미달 ERROR), eval_lines, board_check(시간 초과 ERROR, 틀린 줄은 기대값과 함께) |
| 브리지 (`bridge/logs/bridge.log`) | stderr, INFO (stdout은 확장 통로) | DEBUG, 1MB × 4개 로테이션 | 업로드 실패 `logger.exception`, 포트 열기 실패·시리얼 오류 `logger.exception`, 시간 초과·보드 오류 응답 WARNING, 요청 토큰 수·캐시 사용·보드의 JSON 아닌 출력 DEBUG |
| 펌웨어 | | | 오류 응답에 이유(`model load: ESP_ERR_...`, `bad request: no id` 등)와 아는 경우 요청 id를 넣는다. id가 없으면 브리지가 30초 시간 초과까지 기다렸다 |

- 형식은 수업대로 `시각 레벨 로거이름 파일:줄 메시지`. 브리지는 세션 id를 함께 남겨 `events.jsonl`·DB 행과 이어 볼 수 있게 했다.
- `bridge/logs/events.jsonl`은 DB에 올릴 데이터라 로테이션하지 않는다. 진단 로그(`bridge.log`)만 로테이션한다.
- 이 LOG의 숫자는 이제 로그 파일에서 옮기고 파일 경로를 함께 적는다. 시각은 로그 파일 시각이나 `date` 결과만 쓴다.
- 확인: `eval_bin.py` → `train/logs/eval_bin-20261103-132533.log` (1.000, 51/51, 기준 0.95 통과). `export.py` → `train/logs/export-20261103-132548.log` (torch 대비 2.71e-04, 힐링 없음 0.961, 있음 1.000, 이전과 같음). 브리지 실행 → `bridge/logs/bridge.log`에 COM9 열기 실패가 ERROR와 Traceback으로 남음 (VS Code 브리지가 포트 사용 중).
- 시행착오: Bash heredoc 안의 Python 치환 스크립트에서 `\n`이 `\n`으로 바뀌어 문자열이 맞지 않았다. 파일 수정은 Edit 도구로 했다.
- 펌웨어는 빌드만 했고 아직 굽지 않았다.

## 2026-11-03 13:43 제안의 문법 검사 (codeop)

수락한 제안이 맞는 코드였는지 알 수 없어서, 브리지가 제안마다 '앞부분 + 완성'의 문법을 검사해 DB에 남기게 했다.

- 직접 규칙을 짜지 않고 표준 라이브러리 `codeop.compile_command(src, symbol="exec")`를 쓴다. 대화형 Python이 `...`(계속 입력)을 띄울지 오류를 낼지 정하는 모듈이라, 덜 쓴 코드(`incomplete`)와 틀린 코드(`error`)를 Python 파서가 직접 나눈다.
- 미리 확인 (Python 3.12.14, 9가지): `print("Hello, World!")` ok, `for ...:` 직후·`sum(`·`if ... else:` 중간 incomplete, 빈 줄 섞인 여러 문장 ok, `for i in range(1, 10);` error `invalid syntax`, 실사용 반복 출력 `af" = " = ")` error `unmatched ')'`, 이유 없는 들여쓰기 error `unexpected indent`, `print("hello` error `unterminated string literal`.
- 분류: `ok`, `incomplete`, `error`, `prefix_error`(커서 줄 앞의 완성된 줄들이 이미 오류, 모델 탓이 아님), 비움(앞부분이 1024자로 잘려 파일 앞쪽이 없어 판단 불가). 오류 메시지는 `syntax_error`.
- DB: Supabase 마이그레이션 `events_add_syntax`로 `syntax`(check 제약), `syntax_error` 칸 추가. 브리지보다 먼저 적용했다 (칸이 없으면 업로드가 계속 실패한다). `db/schema.sql`에 칸과 성적표의 `syntax_error_pct`(전체 제안), `accepted_syntax_error_pct`(수락한 제안)를 추가. 분모는 판단 가능한 제안(ok·incomplete·error)만.
- 브리지 테스트 15건 통과 (문법 검사 8가지 추가).
- 실사용 확인은 아직. VS Code가 띄운 브리지(세션 `9351c8a5`)는 이 변경 전에 실행된 것이라 문법 칸을 남기지 않는다.

## 2026-11-03 13:53 DB를 세 테이블로 나누고 판단용 지표·버전 추가

점검 (13:45, 기존 `events` 한 테이블): 제안 79건 중 빈 완성 21건(26.6%)인데 원인(EOS, 힐링 불일치, 공백뿐)을 나눌 칸이 없었다. 어휘에 없어 버린 입력 토큰 수, 입력 토큰 수, 화면에 실제로 보여 준 제안 수, 펌웨어·브리지 버전이 없었다. `model_version`은 펌웨어 상수(`pythia-14m-ft-1`)라 `model.bin`만 바꿔 구우면 틀리게 기록된다. accept 행은 13칸 중 3칸만 쓰였다.

바꾼 것:
- DB 마이그레이션 `split_sessions_suggestions_feedback`: `sessions`(브리지 실행 1회), `suggestions`(제안 1건), `feedback`(shown·accept), 뷰 `scorecard`·`stop_reasons`(security_invoker). 기존 `events`는 남기고 데이터를 옮겼다 (세션·제안 82건, 수락 9건). Supabase security advisor 경고 없음.
- 버전 세 가지, 모두 자동: `model_version`은 export.py가 `<체크포인트 이름>-<model.bin 내용 sha256 앞 8자리>`로 정해 model.bin 끝과 `bridge/vocab.json`에 쓴다 (지금 `pythia-14m-ft-0bbd3425`). `fw_version`은 ESP-IDF가 빌드 때 `git describe`로 넣는 값(`589d8d3-dirty`)과 ELF 해시 앞 8자리(`fw_build`). `bridge_version`은 브리지 실행 때 `git describe --always --dirty`.
- 펌웨어: `{"id":0,"info":true}`에 버전으로 답한다 (브리지는 부팅이 끝난 보드에 붙어 부팅 메시지를 못 보므로 직접 묻는다). 응답마다 `stop`(eos, newline, max_tokens, heal_mismatch, blank, seq_full, empty_input).
- 브리지: 입력 토큰 수와 버린 토큰 수, 캐시 여부, 멈춘 이유를 남긴다. 시작 때 보드 모델 버전과 vocab.json 버전이 다르면 오류를 내고 끝난다. 기록 파일은 `logs/rows.jsonl`(테이블 표시), 업로드는 세션 → 제안 → 반응 순서.
- 확장: VS Code에 넘긴 제안마다 `shown`을 보낸다 (수락률의 분모).
- 확인: 브리지 테스트 16건, 확장 통합 테스트 1건(shown → accept 순서) 통과, 펌웨어 빌드 성공, export 다시 실행 (`train/logs/export-20261103-134932.log`, 정확도 그대로). 확장 다시 설치.
- 시행착오: 통합 테스트에서 가짜 브리지가 Windows에서 `\r\n`으로 써서 문자열 비교가 실패했다. 줄 단위 비교로 고쳤다. heredoc 안 Python 치환에서 `\n`이 실제 줄바꿈으로 바뀌어 TypeScript 문법 오류가 났다 (13:26 항목과 같은 원인), Edit 도구로 고쳤다.
- 보드에는 아직 굽지 않았다. 펌웨어와 model.bin(끝에 버전이 붙음)을 둘 다 다시 구워야 한다.

## 2026-11-03 13:59 새 지표 첫 실사용 확인, 문법 판정 규칙 수정

굽기: 펌웨어 + model.bin (사용자). 세션 `1554662f` (13:56~13:57, playground `01_hello.py`에서 일부러 엉뚱한 제안도 수락).

`sessions`: bridge `589d8d3-dirty`, fw `589d8d3-dirty` (빌드 `a01c495b`), model `pythia-14m-ft-0bbd3425`, 어휘 152. 굽기 전에 뜬 세션 `92f6ee62`는 예전 펌웨어라 버전 요청에 답이 없어 fw·model이 비어 있다 (의도대로).

`scorecard` (이 버전 조합): 제안 27 (캐시 2), 화면 표시 14, 수락 11 (표시 대비 78.6%), 빈 완성 33.3%, 평균 지연 725 ms, 평균 입력 32토큰, 입력 버림 0.8%, 문법 오류율 84.2% (수락한 것 88.9%), 최소 힙 2,113,156.
`stop_reasons`: newline 15, blank 8 (전부 빈 완성), max_tokens 2 (`(f"2 * op(f"2 * op...` 반복).

문법 판정 실수 두 가지 (84.2%는 부풀려진 값):
1. 빈 완성도 판정해서, 앞부분 자체의 오류가 `error`로 잡혔다 (req 6, 9, 11 등).
2. `prefix_error`가 커서 줄 앞의 줄만 봤다. req 8 수락으로 커서 줄이 `print("Hello, World!") in {name}.")`(따옴표 안 닫힘)가 된 뒤, 같은 줄에 붙인 완성이 모두 `unterminated string literal (line 2)`로 모델 탓이 되었다.

조치: 빈 완성은 판단하지 않는다(비움). '앞부분 + 완성'의 오류가 앞부분만 넣었을 때와 같은 오류(메시지·줄)면 `prefix_error`. 실사용 예 4가지를 테스트에 추가 (브리지 테스트 20건 통과). 이 세션 27건을 새 규칙으로 다시 판정 (`prefix_tail`이 끝 200자뿐이라 앞부분이 그보다 긴 몇 건은 정확하지 않음): ok 3→2, error 16→3, prefix_error 8→13, 판단 안 함 0→9.

또 하나: 브리지 코드를 고쳐도 `bridge_version`이 `589d8d3-dirty`로 같아 고치기 전후가 섞인다. 커밋 안 된 변경이 있으면 `bridge.py` 내용 해시를 붙인다 (지금 `589d8d3-dirty+9f3ac12d`). 이미 DB에 들어간 `589d8d3-dirty` 행의 문법 값은 고치지 않고 두며, 위 이유로 부풀려진 값이다.

## 2026-11-03 14:09 범용성 작업: 기준 측정, 어휘 크기 결정

설계: 데이터에 MBPP(초급 Python 974문제, CC-BY-4.0, `train/data_ext/mbpp.jsonl`)를 더하고, 어휘에 없는 토큰은 버리지 않고 어휘 안의 더 작은 토큰(최악은 1바이트)으로 쪼갠다. 원본 Pythia-14M에서 시작해 학습 전에 어휘를 줄인다.

- 데이터 (`train/corpus.py`): MBPP 문제 설명을 첫 줄 주석으로 붙이고 `\r\n`→`\n`, 탭→공백 4칸, 줄 끝 공백 제거. 정리 후 문법 오류인 문제는 뺀다. 평가 전용 heldout = task_id 411~510 (100문제), 나머지 874문제 학습.
- 쪼개기 규칙은 `bridge/vocab_map.py` 하나를 학습·평가·브리지가 함께 쓴다.
- 시행착오: Pythia 토크나이저의 1바이트 토큰은 256개가 아니라 243개다. 빠진 13개(0xC0, 0xC1, 0xF5~0xFF)는 UTF-8 문자열에 나올 수 없는 바이트라 문제없다. 검사를 그에 맞게 고쳤다.

어휘 크기 측정 (학습 코퍼스 75,279토큰, 서로 다른 토큰 2,560개; 자주 나오는 K개 + 1바이트 243개 + EOS; heldout 8,888토큰 기준):

| K | 어휘 | heldout 길이 증가 | 쪼개진 토큰 | model.bin |
|---|---|---|---|---|
| 256 | 500 | +39.6% | 13.2% | 5.27MB |
| 512 | 756 | +21.7% | 8.5% | 5.53MB |
| 768 | 1,012 | +13.6% | 5.9% | 5.80MB |
| 1,024 | 1,268 | +9.9% | 4.8% | 6.06MB |

결정: K=768 (어휘 1,012). 1,024는 길이 이득 3.7%p에 비해 PSRAM(모델 + KV 캐시 1.5MB)이 8MB에 너무 가깝다.

기준 (지금 모델 `pythia-14m-ft-0bbd3425`, `train/logs/eval_bin-20261103-140849.log`, 문법 판정은 `bridge/syntax.py`):

| 세트 | 줄 | 정확 일치 | 문법 통과 (ok+incomplete / 판단 가능) | 입력 토큰 버림 |
|---|---|---|---|---|
| examples | 51 | 1.000 | 1.000 | 0.0% |
| heldout | 700 | 0.000 | 0.629 (ok 252, incomplete 176, error 252, 판단 안 함 20) | 47.9% |

## 2026-11-03 14:14 범용 학습 (train_general.py) → heldout 문법 0.629 → 0.912

`uv run train_general.py --k 768 --steps N` : 원본 Pythia-14M을 float32로 불러 embed_in·embed_out을 어휘 1,012행으로 자른 뒤 학습. MBPP 874문제 + 예시 8개×8 = 938시퀀스, 88,864토큰. 배치 32, AdamW lr 5e-4 (OneCycle), weight decay 0.01, 시퀀스 최대 256. heldout loss가 가장 낮은 스텝을 저장. 3060에서 1,500스텝 약 1분.

| 실행 | heldout loss (학습 전 3.1606) | 로그 |
|---|---|---|
| 1,500스텝 | 100: 1.8261, **200: 1.6650**, 300: 1.7593, 500: 2.0178, 1000: 2.3739, 1500: 2.4169 | `train/logs/train_general-20261103-141121.log` |
| 300스텝 | 100: 1.6800, 200: 1.5556, **225: 1.5540**, 300: 1.5647 | `train/logs/train_general-20261103-141300.log` |

- 1,500스텝은 200스텝(약 7에폭) 뒤부터 과적합. 최저점이 학습률이 높은 구간이라, 학습률이 끝까지 내려가는 300스텝 일정이 더 낮았다(1.5540). 300스텝 225스텝 모델을 쓴다.
- heldout loss로 저장 시점을 골랐으므로 heldout 평가에 약간 유리하게 기운다.
- 시행착오: transformers 5에서 출력층 속성이 `embed_out`이 아니라 `AttributeError`. `get/set_output_embeddings()`로 바꿨다 (저장 파일 키는 여전히 `embed_out.weight`). 20스텝 시험 실행에서 OneCycle 워밍업이 0스텝이 되어 `ZeroDivisionError`, 워밍업을 최소 2스텝으로.

내보내기 `uv run export.py ckpt/pythia-14m-general-k768-s300` → 모델 `pythia-14m-general-k768-s300-40b80eb5`, model.bin 5.8MB급, torch 대비 로짓 최대 오차 4.68e-04, argmax 일치 (27토큰). `npmodel.py`는 1바이트 토큰(UTF-8 글자 일부) 때문에 토큰을 바이트로 다루게 바꿨다.

평가 `eval_bin.py` (`train/logs/eval_bin-20261103-141353.log`):

| 세트 | 지표 | 기준 `pythia-14m-ft-0bbd3425` | 새 `pythia-14m-general-k768-s300-40b80eb5` |
|---|---|---|---|
| examples 51줄 | 정확 일치 | 1.000 | 1.000 |
| heldout 700줄 | 정확 일치 | 0.000 | 0.117 (82줄) |
| heldout | 문법 통과 | 0.629 | 0.912 (ok 297, incomplete 323, error 60, 판단 안 함 20) |
| heldout | 입력 토큰 | 47.9% 버림 | 버림 0, 7.6% 쪼갬 (입력 손실 없음) |
| examples | 입력 토큰 | 버림 0 | 16.5% 쪼갬 (예시 고유 문자열이 상위 768개에 안 듦) |

## 2026-11-03 14:16 보드·브리지·CI를 새 모델에 맞춤

- 펌웨어: 어휘 상한 `MAX_V` 512 → 1,280. 토큰 문자열 자리를 '어휘 × 256바이트'(1,012개면 약 259KB)에서 16KB 고정(`STR_MAX`, 실제 약 6KB)으로 줄이고 넘치면 `ESP_ERR_INVALID_SIZE`. PSRAM 예상: 모델 약 5.8MB + KV 캐시 1.5MB. 빌드 성공.
- 브리지: vocab.json에 `fallback: true`면 `bridge/vocab_map.py`로 쪼갠다 (학습·평가와 같은 함수). `suggestions.fallback_tokens` 칸 추가 (마이그레이션 `suggestions_add_fallback_tokens`), `scorecard`에 `fallback_token_pct`. 1바이트 토큰 때문에 보드가 UTF-8 글자를 반만 보낼 수 있어, JSON 해석 전에 깨진 바이트를 대체 문자로 바꾼다 (그대로면 해석 실패 → 30초 시간 초과).
- 브리지 테스트 21건 통과 (버리기·쪼개기 두 방식 각각, 쪼개도 입력 바이트가 그대로 남는지).
- CI `model-eval`: `--min-examples 0.95 --min-heldout-syntax 0.85` (지금 1.000, 0.912). 로컬 통과.
- 시행착오: 20스텝 시험 실행 로그 4개(`train_general-20261103-1410*.log`, 위 AttributeError·ZeroDivisionError가 난 실행)를 정리하다 지웠다. 오류 내용은 14:14 항목에 적혀 있다.
- 보드에는 아직 굽지 않았다 (펌웨어 + model.bin).

## 2026-11-03 14:21 새 모델 보드 측정: 51/51, 평균 1,117 ms

굽기 (사용자, 14:19 무렵): bootloader 21,168 B, 파티션 표 3,072 B, 펌웨어 204,240 B (1.2초), model.bin `pythia-14m-general-k768-s300-40b80eb5` 5,800,616 B at 0x190000 (41.0초). 넷 다 `Hash of data verified`. 처음 명령은 `0x190000` 뒤 파일 경로가 빠져 esptool이 `Must be pairs of an address and the binary filename`으로 아무것도 쓰지 않았다.

`uv run board_check.py` (사용자 실행, `train/logs/board_check-20261103-142020.log`):

| | 기존 `pythia-14m-ft-0bbd3425` (13:17) | 새 `pythia-14m-general-k768-s300-40b80eb5` |
|---|---|---|
| 다음 줄 정확도 (예시 51줄) | 1.000 | 1.000 |
| 평균 생성 시간 | 760 ms | 1,117 ms (+47%) |
| 토큰당 (앞부분 처리 포함) | 83 ms | 97 ms |
| 최소 남은 힙 (PSRAM 포함) | 2,113,284 B | 1,188,740 B |
| 첫 줄 입력·생성 토큰 | 3·9 | 10·17 |
| 마지막 줄 입력 토큰 | 130 | 176 |

- 모델 로드 성공 (model.bin 5.8MB + KV 캐시 1.5MB가 PSRAM에 들어감). 남은 힙 약 1.19MB.
- 느려진 원인 (숫자로 보이는 것): 예시의 토큰 16.5%가 쪼개져 입력·생성 토큰 수가 늘었다 (첫 줄 입력 3 → 10, 생성 9 → 17). 어휘가 152 → 1,012라 출력층 계산(어휘 × 128)도 늘어 토큰당 83 → 97 ms.
