// USB 시리얼로 JSON 한 줄 요청을 받아 JSON 한 줄로 답한다.
//   완성: {"id":1,"tokens":[...],"heal":"..."}
//      → {"id":1,"completion":"...","stop":"newline","gen_ms":0,"tokens":0,"heap_min":0}
//   버전: {"id":0,"info":true}
//      → {"id":0,"fw_version":"<git describe>","fw_build":"<ELF sha256 앞 8자리>","model_version":"...","vocab":152}
// 토큰화(줄인 어휘 id)와 토큰 힐링 문자열은 브리지가 만든다.
#include <stdio.h>
#include <string.h>
#include "cJSON.h"
#include "driver/usb_serial_jtag.h"
#include "esp_app_desc.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "model.h"

#define MAX_NEW 32

static const char *TAG = "main";

static void reply(const char *s)
{
    usb_serial_jtag_write_bytes(s, strlen(s), portMAX_DELAY);
    usb_serial_jtag_write_bytes("\n", 1, portMAX_DELAY);
}

static void reply_json(cJSON *res)
{
    char *out = cJSON_PrintUnformatted(res);
    reply(out);
    cJSON_free(out);
    cJSON_Delete(res);
}

// 오류 응답. id를 알면 넣어서 브리지가 시간 초과까지 기다리지 않게 한다
static void reply_error(const cJSON *id, const char *why)
{
    ESP_LOGW(TAG, "%s", why);
    cJSON *res = cJSON_CreateObject();
    if (cJSON_IsNumber(id)) cJSON_AddNumberToObject(res, "id", id->valuedouble);
    cJSON_AddStringToObject(res, "error", why);
    reply_json(res);
}

// 펌웨어 버전은 빌드 때 ESP-IDF가 git describe로 넣는다 (커밋 안 된 변경이 있으면 -dirty).
// -dirty면 같은 이름으로 다른 빌드가 나올 수 있어 ELF 해시를 함께 준다
static void reply_info(const cJSON *id, esp_err_t err, const char *load_err)
{
    char elf[9];
    esp_app_get_elf_sha256(elf, sizeof elf);
    cJSON *res = cJSON_CreateObject();
    cJSON_AddNumberToObject(res, "id", id->valuedouble);
    cJSON_AddStringToObject(res, "fw_version", esp_app_get_description()->version);
    cJSON_AddStringToObject(res, "fw_build", elf);
    cJSON_AddStringToObject(res, "model_version", model_version());
    cJSON_AddNumberToObject(res, "vocab", err == ESP_OK ? model_vocab() : 0);
    if (err != ESP_OK) cJSON_AddStringToObject(res, "error", load_err);
    reply_json(res);
}

void app_main(void)
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
    esp_err_t err = model_load();
    static char load_err[64];
    snprintf(load_err, sizeof load_err, "model load: %s", esp_err_to_name(err));
    ESP_LOGI(TAG, "fw %s, model %s", esp_app_get_description()->version, model_version());
    if (err == ESP_OK) reply("{\"ready\":true}");
    else {
        ESP_LOGE(TAG, "%s", load_err);
        reply_error(NULL, load_err);
    }

    static char line[4096];
    size_t len = 0;
    uint8_t c;
    for (;;) {
        if (usb_serial_jtag_read_bytes(&c, 1, portMAX_DELAY) != 1) continue;
        if (c != '\n') {
            if (len < sizeof line - 1) line[len++] = (char)c;  // 넘치면 잘려서 아래 파싱이 실패한다
            continue;
        }
        line[len] = '\0';
        len = 0;

        cJSON *req = cJSON_Parse(line);
        const cJSON *id = cJSON_GetObjectItem(req, "id");
        const cJSON *toks = cJSON_GetObjectItem(req, "tokens");
        const cJSON *heal = cJSON_GetObjectItem(req, "heal");
        if (cJSON_IsNumber(id) && cJSON_IsTrue(cJSON_GetObjectItem(req, "info"))) {
            reply_info(id, err, load_err);  // 모델 로드에 실패해도 버전은 답한다
            cJSON_Delete(req);
            continue;
        }
        const char *why = !req ? "bad request: not JSON"
                          : err != ESP_OK ? load_err
                          : !cJSON_IsNumber(id) ? "bad request: no id"
                          : !cJSON_IsArray(toks) ? "bad request: no tokens" : NULL;
        if (why) {
            reply_error(id, why);
            cJSON_Delete(req);
            continue;
        }
        int64_t t0 = esp_timer_get_time();
        static int ids[1024];
        int n = 0;
        const cJSON *it;
        cJSON_ArrayForEach(it, toks) if (n < 1024) ids[n++] = it->valueint;
        static char completion[256];
        const char *stop = "";
        int made = model_generate(ids, n, cJSON_IsString(heal) ? heal->valuestring : "", MAX_NEW, completion,
                                  sizeof completion, &stop);
        cJSON *res = cJSON_CreateObject();
        cJSON_AddNumberToObject(res, "id", id->valuedouble);
        cJSON_AddStringToObject(res, "completion", completion);
        cJSON_AddStringToObject(res, "stop", stop);
        cJSON_AddStringToObject(res, "model_version", model_version());
        cJSON_AddNumberToObject(res, "gen_ms", (double)((esp_timer_get_time() - t0) / 1000));
        cJSON_AddNumberToObject(res, "tokens", made);
        cJSON_AddNumberToObject(res, "heap_min", esp_get_minimum_free_heap_size());
        reply_json(res);
        cJSON_Delete(req);
    }
}
