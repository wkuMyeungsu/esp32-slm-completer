# ESP32에서 언어 모델을 돌리는 다른 오픈소스와 비교

조사일 2026-11-03. 각 프로젝트의 README·모델 카드에 적힌 내용만 옮겼고, 적혀 있지 않은 항목은 "없음"으로 표시했다. 숫자는 모델·양자화·클럭이 서로 달라 그대로 맞비교할 수는 없다.

## 비교 대상

| 프로젝트 | 용도 | 링크 |
|---|---|---|
| esp32-slm-completer (이 저장소) | VS Code에서 Python 코드 자동완성 | |
| doryiii/esp32-llm | 동화 생성 (TinyStories) | https://github.com/doryiii/esp32-llm |
| jstephencorey/esp32-llm (DaveBben/esp32-llm 포크) | 동화 생성 (TinyStories) | https://github.com/jstephencorey/esp32-llm |
| slvDev/esp32-ai-barista | 에스프레소 문제 해결 질의응답 | https://huggingface.co/slvDev/esp32-ai-barista |

같은 조사에서 eric-humane/esp32-llm (llama2.c, 260K, 약 32.83 tok/s, ESP-DSP SIMD), ruvllm-esp32 (Rust, int8, 260K급에서 20~50 tok/s, 어휘 128)도 나왔다. 코드 자동완성을 보드에서 직접 하는 프로젝트는 찾지 못했다.

## 사실 비교

| | 이 저장소 | doryiii/esp32-llm | jstephencorey/esp32-llm | slvDev/esp32-ai-barista |
|---|---|---|---|---|
| 바탕 | Pythia-14M (GPT-NeoX), 추론 C 코드 직접 작성 | Karpathy llama2.c (`run.c`, `runq.c`) | llama2.c | 자체 구조 6층, d_model 128, 4헤드 |
| 파라미터 | 약 130만 (층 약 119만 + 입출력 임베딩 2 × 1,012 × 128) | 260K, 3.3M | 260K | 8,903,808 (핵심 1,575,424) |
| 어휘 | 입력·출력 모두 1,012 (자주 나오는 768 + 1바이트 243 + EOS) | 없음 | 없음 | 입력 8,057 / 출력 854 |
| 수치 형식 | float32 | float32 (`llm.c`), int8 (`llm8.c`) | 없음 | int4 저장(그룹 128, 약 4.6MB) → 부팅 때 int8, 활성값도 int8 |
| 가중치 위치 | 플래시 `model` 파티션 → 부팅 때 전부 PSRAM | SPIFFS(플래시) | 없음 | 큰 임베딩은 플래시 mmap, 핵심·출력층은 PSRAM |
| KV 캐시 | PSRAM, 최대 256위치, 이전 요청과 겹치는 앞부분 재사용 | 없음 | 없음 | PSRAM, 문맥 128 |
| 토크나이저 | PC(브리지, `bridge/vocab_map.py`) | 보드 | 없음 | 보드 (ASCII만) |
| 연산 가속 | 일반 C 반복문 | int8은 Xtensa PIE SIMD, f32는 esp-dsp, 활성값은 내부 RAM | 듀얼 코어, esp-dsp 내적 | 없음 |
| 클럭·캐시 | CPU 240MHz, PSRAM 80MHz, 데이터 캐시 64KB·라인 64B | 코어·RAM·SPI 최대, 데이터 캐시 64KB | CPU 240MHz, PSRAM 80MHz, 명령 캐시 확대 | 없음 |
| 속도 | 토큰당 97ms (앞부분 처리 포함, `train/LOG.md` 14:21) | 260K f32 약 30 tok/s, 3.3M int8 약 12 tok/s | 19.13 tok/s | 16.60 조각/s (60.25ms, 시리얼만) |
| 병목 판단 | PSRAM 읽기 (설정 변경만으로 한 스텝 147 → 64ms, LOG 12:29) | 메모리 대역폭 ("한계 13.1 tok/s에 매우 가까움"), 그래서 두 번째 코어 안 씀 | 없음 | 없음 |
| 보드 | ESP32-S3, PSRAM 8MB, 플래시 8MB | Waveshare ESP32-S3-LCD-1.9, PSRAM 2MB(260K)·8MB(3.3M) | ESP32-S3FH4R2, PSRAM 2MB | PSRAM 8MB, 내부 SRAM 512KB |
| 정확도 평가 | 학습에 안 쓴 MBPP 100문제(700줄) + 예시 51줄, CI 기준 (예시 0.95, heldout 문법 통과 0.85) | 없음 | 없음 | 공개 수치 없음. 평가 세트가 모델 선택에 쓰여 분리되어 있지 않다고 적혀 있음 |
| 범위 제한 | 초급 Python (MBPP + 예시 8개) | 동화 | 동화 | 에스프레소 문제 해결만. 범위 밖 질문 거절을 학습으로 넣음. 출력 어휘(854) 밖의 글자는 낼 수 없음 |

## 차이에서 본 더 좋았을 방법

- **양자화**: 다른 프로젝트는 int8·int4로 토큰마다 읽는 양을 줄였다. 이 저장소는 float32라 토큰마다 약 4.8MB를 PSRAM에서 읽는다. int8이면 읽는 양이 1/4이라, 같은 속도로 약 4배 큰 모델(예: Pythia-31M, float32 약 19MB → int8 약 5MB)을 올릴 수 있었다. 정확도를 올리는 가장 직접적인 길이다.
- **입력·출력 어휘 분리**: barista는 입력 어휘(8,057)와 출력 어휘(854)를 따로 둔다. 입력 임베딩은 토큰마다 한 행만 읽으므로 커도 싸고 플래시 mmap에 둘 수 있다. 출력층은 토큰마다 전부 계산하므로 작게 둔다. 이 저장소는 둘 다 1,012라, 입력이 쪼개져(대표 예시 토큰의 16.5%) 앞부분 토큰 수가 늘고 속도도 떨어졌다 (평균 760 → 1,117ms, LOG 14:21).
- **범위 좁히기**: barista는 한 주제로 범위를 좁히고 범위 밖 거절까지 학습시켰다. 이 저장소의 학습 데이터는 MBPP 함수 문제(첫 줄 `# Write a function ...`)가 938시퀀스 중 874개라, playground에서 쓰는 스크립트형 코드(`print`, `input`)와 어긋난다. 주석 없이 `print`부터 치면 ` Write Write Write function ...`이 나왔다 (실사용 로그 14:25).
- **연산 가속**: 다른 프로젝트는 esp-dsp나 PIE SIMD를 쓴다. 병목이 메모리 대역폭이라는 판단(doryiii)이 맞다면 float32에서는 효과가 작고, int8과 함께 쓸 때 의미가 있다.
- **이 저장소가 다른 점**: 분리된 평가 세트와 CI 기준으로 정확도를 관리하고, 실사용 기록을 모델·펌웨어·브리지 버전별로 DB에 남긴다. 다른 세 프로젝트에는 공개된 정확도 평가가 없다.

## 출처

- doryiii/esp32-llm README: https://github.com/doryiii/esp32-llm
- jstephencorey/esp32-llm README: https://github.com/jstephencorey/esp32-llm
- eric-humane/esp32-llm: https://github.com/eric-humane/esp32-llm
- slvDev/esp32-ai-barista 모델 카드: https://huggingface.co/slvDev/esp32-ai-barista
- ruvllm-esp32: https://docs.rs/ruvllm-esp32
- Circuit Digest, Running 260K Tiny-LLM on ESP32: https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32
