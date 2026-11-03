#pragma once
#include <stddef.h>
#include "esp_err.h"

#define MODEL_MAX_SEQ 256
#define MODEL_MAX_NEW 64  // 한 번에 생성할 최대 토큰

// 플래시 "model" 파티션에서 model.bin을 읽어 PSRAM에 올린다
esp_err_t model_load(void);

// model.bin 끝에 적힌 모델 버전 (없으면 "unknown"), 어휘 크기
const char *model_version(void);
int model_vocab(void);

// ids(줄인 어휘)를 앞부분으로 탐욕적 생성. heal은 토큰 힐링 문자열(없으면 "").
// out에 heal을 잘라 낸 한 줄 완성을 쓰고, 생성한 토큰 수를 돌려준다.
// stop: 멈춘 이유 eos · newline · max_tokens · repeat · heal_mismatch · blank · seq_full · empty_input
int model_generate(const int *ids, int n, const char *heal, int max_new, char *out, size_t cap, const char **stop);
