// USB 시리얼로 JSON 한 줄 요청 {"id":1,"prefix":"..."} 을 받아
// {"id":1,"completion":"...","model_version":"...","gen_ms":0,"tokens":0,"heap_min":0} 로 답한다.
#include <string.h>
#include "cJSON.h"
#include "driver/usb_serial_jtag.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"

#define MODEL_VERSION "stub"

static void reply(const char *s)
{
    usb_serial_jtag_write_bytes(s, strlen(s), portMAX_DELAY);
    usb_serial_jtag_write_bytes("\n", 1, portMAX_DELAY);
}

// shortcut: 고정 문자열을 돌려준다, SLM 추론이 들어오면 교체
static const char *complete(const char *prefix)
{
    (void)prefix;
    return "  # slm stub";
}

void app_main(void)
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
    reply("{\"ready\":true,\"model\":\"stub\"}");

    static char line[4096];
    size_t n = 0;
    uint8_t c;
    for (;;) {
        if (usb_serial_jtag_read_bytes(&c, 1, portMAX_DELAY) != 1) continue;
        if (c != '\n') {
            if (n < sizeof line - 1) line[n++] = (char)c;  // 넘치면 잘려서 아래 파싱이 실패한다
            continue;
        }
        line[n] = '\0';
        n = 0;

        cJSON *req = cJSON_Parse(line);
        const cJSON *id = cJSON_GetObjectItem(req, "id");
        const cJSON *prefix = cJSON_GetObjectItem(req, "prefix");
        if (!cJSON_IsNumber(id) || !cJSON_IsString(prefix)) {
            reply("{\"error\":\"bad request\"}");
            cJSON_Delete(req);
            continue;
        }
        int64_t t0 = esp_timer_get_time();
        const char *completion = complete(prefix->valuestring);
        cJSON *res = cJSON_CreateObject();
        cJSON_AddNumberToObject(res, "id", id->valuedouble);
        cJSON_AddStringToObject(res, "completion", completion);
        cJSON_AddStringToObject(res, "model_version", MODEL_VERSION);
        cJSON_AddNumberToObject(res, "gen_ms", (double)((esp_timer_get_time() - t0) / 1000));
        cJSON_AddNumberToObject(res, "tokens", 0);  // shortcut: 추론이 들어오면 실제 생성 토큰 수
        cJSON_AddNumberToObject(res, "heap_min", esp_get_minimum_free_heap_size());
        char *out = cJSON_PrintUnformatted(res);
        reply(out);
        cJSON_free(out);
        cJSON_Delete(res);
        cJSON_Delete(req);
    }
}
