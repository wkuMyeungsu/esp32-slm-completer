// GPT-NeoX(Pythia) 한 토큰씩 추론. train/export.py의 NpModel을 그대로 옮겼다.
// 가중치는 플래시 "model" 파티션의 model.bin을 PSRAM으로 읽어 쓴다 (형식은 export.py 설명 참고).
#include "model.h"

#include <math.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MAGIC 0x314D4C53
#define MAX_L 8
#define MAX_D 256
#define MAX_F 1024
#define MAX_V 1280  // 어휘 상한 (지금 1,012)
#define STR_MAX 16384  // model.bin의 토큰 문자열 + 모델 버전 최대 바이트
#define REPEAT 3       // 같은 조각이 3번 연속이면 반복으로 보고 멈춘다 (npmodel.py REPEAT와 같게)
#define MAX_PERIOD 8   // 반복 조각 최대 길이 (토큰)

static const char *TAG = "model";

typedef struct {
    const float *ln1w, *ln1b, *ln2w, *ln2b, *qkvw, *qkvb, *ow, *ob, *f1w, *f1b, *f2w, *f2b;
} layer_t;

static int D, L, H, F, V, ROT, EOS, HD;
static char mver[64] = "unknown";
static const float *emb, *lnfw, *lnfb, *outw;
static layer_t lay[MAX_L];
static const char *tstr[MAX_V];
static uint8_t tlen[MAX_V];
static float *kc, *vc;  // [L][MODEL_MAX_SEQ][D], PSRAM
static float x[MAX_D], h[MAX_D], qkv[3 * MAX_D], att[MAX_D], ao[MAX_D], mo[MAX_D], mf[MAX_F], logits[MAX_V];
static float sc[MODEL_MAX_SEQ];
static int cached[MODEL_MAX_SEQ], ncached;  // KV 캐시에 들어 있는 위치별 토큰 (다음 요청에서 겹치는 앞부분은 다시 계산하지 않는다)

static const float *take(const float **p, int n)
{
    const float *r = *p;
    *p += n;
    return r;
}

esp_err_t model_load(void)
{
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) return ESP_ERR_NOT_FOUND;
    int32_t hd[8];
    ESP_ERROR_CHECK(esp_partition_read(part, 0, hd, sizeof hd));
    if (hd[0] != MAGIC) return ESP_ERR_INVALID_VERSION;
    D = hd[1], L = hd[2], H = hd[3], F = hd[4], V = hd[5], ROT = hd[6], EOS = hd[7];
    HD = D / H;
    if (D > MAX_D || L > MAX_L || F > MAX_F || V > MAX_V) return ESP_ERR_INVALID_SIZE;

    size_t nf = (size_t)2 * V * D + 2 * D + (size_t)L * (4 * D + 3 * D * D + 3 * D + D * D + D + F * D + F + D * F + D);
    // 토큰 문자열 + 모델 버전 자리. 어휘 1,012개에 약 6KB. 토큰마다 최대(255바이트)로 잡으면 PSRAM을 260KB 더 쓴다
    size_t nbytes = sizeof hd + nf * 4 + STR_MAX;
    if (nbytes > part->size) nbytes = part->size;
    uint8_t *buf = heap_caps_malloc(nbytes, MALLOC_CAP_SPIRAM);
    kc = heap_caps_malloc((size_t)L * MODEL_MAX_SEQ * D * 4, MALLOC_CAP_SPIRAM);
    vc = heap_caps_malloc((size_t)L * MODEL_MAX_SEQ * D * 4, MALLOC_CAP_SPIRAM);
    if (!buf || !kc || !vc) return ESP_ERR_NO_MEM;
    ESP_ERROR_CHECK(esp_partition_read(part, 0, buf, nbytes));

    const float *p = (const float *)(buf + sizeof hd);
    emb = take(&p, V * D);
    for (int l = 0; l < L; l++) {
        layer_t *y = &lay[l];
        y->ln1w = take(&p, D), y->ln1b = take(&p, D), y->ln2w = take(&p, D), y->ln2b = take(&p, D);
        y->qkvw = take(&p, 3 * D * D), y->qkvb = take(&p, 3 * D);
        y->ow = take(&p, D * D), y->ob = take(&p, D);
        y->f1w = take(&p, F * D), y->f1b = take(&p, F);
        y->f2w = take(&p, D * F), y->f2b = take(&p, D);
    }
    lnfw = take(&p, D), lnfb = take(&p, D);
    outw = take(&p, V * D);
    const uint8_t *s = (const uint8_t *)p, *end = buf + nbytes;
    for (int i = 0; i < V; i++) {
        if (s >= end || s + 1 + *s > end) return ESP_ERR_INVALID_SIZE;  // 토큰 문자열이 STR_MAX를 넘음
        tlen[i] = *s++;
        tstr[i] = (const char *)s;
        s += tlen[i];
    }
    // 토큰 문자열 뒤: uint8 길이 + 모델 버전 문자열 (export.py가 씀, 없으면 unknown)
    if (s < end && *s > 0 && *s < sizeof mver && s + 1 + *s <= end) {
        memcpy(mver, s + 1, *s);
        mver[*s] = '\0';
    }
    ESP_LOGI(TAG, "%s: dim %d, layers %d, heads %d, ffn %d, vocab %d, rot %d, %u bytes", mver, D, L, H, F, V, ROT,
             (unsigned)nbytes);
    return ESP_OK;
}

const char *model_version(void) { return mver; }
int model_vocab(void) { return V; }

static void matvec(float *out, const float *w, const float *b, const float *in, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        const float *r = w + (size_t)i * cols;
        float acc = b ? b[i] : 0.f;
        for (int j = 0; j < cols; j++) acc += r[j] * in[j];
        out[i] = acc;
    }
}

static void layernorm(float *out, const float *in, const float *w, const float *b)
{
    float m = 0, v = 0;
    for (int i = 0; i < D; i++) m += in[i];
    m /= D;
    for (int i = 0; i < D; i++) v += (in[i] - m) * (in[i] - m);
    float r = 1.f / sqrtf(v / D + 1e-5f);
    for (int i = 0; i < D; i++) out[i] = (in[i] - m) * r * w[i] + b[i];
}

static void rope(float *q, int pos)
{
    int half = ROT / 2;
    for (int i = 0; i < half; i++) {
        double ang = pos * pow(10000.0, -2.0 * i / ROT);
        float c = (float)cos(ang), s = (float)sin(ang), x1 = q[i], x2 = q[i + half];
        q[i] = x1 * c - x2 * s;
        q[i + half] = x2 * c + x1 * s;
    }
}

static const float *step(int tok, int pos)
{
    cached[pos] = tok;
    ncached = pos + 1;
    memcpy(x, emb + (size_t)tok * D, D * 4);
    for (int l = 0; l < L; l++) {
        const layer_t *y = &lay[l];
        float *K = kc + (size_t)l * MODEL_MAX_SEQ * D, *Vv = vc + (size_t)l * MODEL_MAX_SEQ * D;
        layernorm(h, x, y->ln1w, y->ln1b);
        matvec(qkv, y->qkvw, y->qkvb, h, 3 * D, D);
        for (int hh = 0; hh < H; hh++) {
            float *q = qkv + hh * 3 * HD, *k = q + HD, *v = q + 2 * HD;
            rope(q, pos);
            rope(k, pos);
            memcpy(K + (size_t)pos * D + hh * HD, k, HD * 4);
            memcpy(Vv + (size_t)pos * D + hh * HD, v, HD * 4);
            float mx = -INFINITY, sum = 0, scale = 1.f / sqrtf((float)HD);
            for (int t = 0; t <= pos; t++) {
                const float *kt = K + (size_t)t * D + hh * HD;
                float a = 0;
                for (int d = 0; d < HD; d++) a += q[d] * kt[d];
                sc[t] = a * scale;
                if (sc[t] > mx) mx = sc[t];
            }
            for (int t = 0; t <= pos; t++) sum += (sc[t] = expf(sc[t] - mx));
            float *o = att + hh * HD;
            memset(o, 0, HD * 4);
            for (int t = 0; t <= pos; t++) {
                const float *vt = Vv + (size_t)t * D + hh * HD;
                float a = sc[t] / sum;
                for (int d = 0; d < HD; d++) o[d] += a * vt[d];
            }
        }
        matvec(ao, y->ow, y->ob, att, D, D);
        layernorm(h, x, y->ln2w, y->ln2b);
        matvec(mf, y->f1w, y->f1b, h, F, D);
        for (int i = 0; i < F; i++) mf[i] = 0.5f * mf[i] * (1.f + erff(mf[i] * 0.70710678f));
        matvec(mo, y->f2w, y->f2b, mf, D, F);
        for (int i = 0; i < D; i++) x[i] += ao[i] + mo[i];
    }
    layernorm(h, x, lnfw, lnfb);
    matvec(logits, outw, NULL, h, V, D);
    return logits;
}

// g 끝이 길이 p(1~MAX_PERIOD)인 조각을 REPEAT번 연속 반복하면 첫 번째 조각까지만 남길 길이, 아니면 -1.
// train/npmodel.py의 repeat_cut과 같은 규칙 (진짜 반복이어도 잘린 앞부분은 맞는 코드라 막지 않고 멈춘다)
static int repeat_cut(const int *g, int n)
{
    for (int p = 1; p <= MAX_PERIOD; p++) {
        int m = p * REPEAT;
        if (n < m) continue;
        bool same = true;
        for (int i = 0; i < m && same; i++) same = g[n - m + i] == g[n - m + i % p];
        if (same) return n - m + p;
    }
    return -1;
}

// 글자가 나온 뒤 줄바꿈이 나왔는지 (완성은 한 줄)
static bool line_done(const char *body)
{
    while (*body == ' ' || *body == '\t' || *body == '\n' || *body == '\r') body++;
    return *body && strchr(body, '\n');
}

int model_generate(const int *ids, int n, const char *heal, int max_new, char *out, size_t cap, const char **stop)
{
    size_t hl = strlen(heal), len = 0;
    char text[512];
    text[0] = '\0';
    out[0] = '\0';
    if (n < 1) {
        *stop = "empty_input";
        return 0;
    }
    if (n > MODEL_MAX_SEQ - max_new) {  // 앞쪽을 버린다
        ids += n - (MODEL_MAX_SEQ - max_new);
        n = MODEL_MAX_SEQ - max_new;
    }
    int pos = 0, made = 0, gen[MODEL_MAX_NEW], ng = 0;  // gen: 힐링이 끝난 뒤 생성한 토큰
    if (max_new > MODEL_MAX_NEW) max_new = MODEL_MAX_NEW;
    while (pos < n - 1 && pos < ncached && cached[pos] == ids[pos]) pos++;  // 이전 요청과 겹치는 앞부분
    for (; pos < n - 1; pos++) step(ids[pos], pos);
    int t = ids[n - 1];
    *stop = "max_tokens";
    for (;;) {
        if (made >= max_new) break;  // stop = max_tokens
        if (pos >= MODEL_MAX_SEQ) {
            *stop = "seq_full";
            break;
        }
        const float *lg = step(t, pos++);
        int best = -1;
        for (int i = 0; i < V; i++) {
            if (len < hl) {  // 힐링: 남은 heal 문자열과 앞이 맞는 토큰만
                size_t rest = hl - len, m = tlen[i] < rest ? tlen[i] : rest;
                if (tlen[i] == 0 || memcmp(tstr[i], heal + len, m) != 0) continue;
            }
            if (best < 0 || lg[i] > lg[best]) best = i;
        }
        made++;
        if (best < 0) {  // heal 문자열과 앞이 맞는 토큰이 하나도 없음
            *stop = "heal_mismatch";
            break;
        }
        if (best == EOS) {
            *stop = "eos";
            break;
        }
        if (len + tlen[best] >= sizeof text) break;  // stop = max_tokens
        bool healed = len >= hl;
        t = best;
        memcpy(text + len, tstr[t], tlen[t]);
        len += tlen[t];
        text[len] = '\0';
        if (healed) {
            gen[ng++] = t;
            int keep = repeat_cut(gen, ng);
            if (keep >= 0) {  // 같은 조각이 REPEAT번 연속: 첫 번째만 남기고 멈춘다
                for (int i = keep; i < ng; i++) len -= tlen[gen[i]];
                text[len] = '\0';
                *stop = "repeat";
                break;
            }
        }
        if (len >= hl && line_done(text + hl)) {
            *stop = "newline";
            break;
        }
        vTaskDelay(1);  // 아이들 태스크에 시간을 줘서 태스크 워치독 경고를 막는다
    }
    if (len < hl || strncmp(text, heal, hl) != 0) {
        if (strcmp(*stop, "eos") != 0) *stop = "heal_mismatch";  // heal을 다 맞추기 전에 끝남
    } else {
        strncpy(out, text + hl, cap - 1);
        out[cap - 1] = '\0';
        char *body = out;  // 글자 뒤의 첫 줄바꿈부터 버린다 (앞쪽 줄바꿈·들여쓰기는 남김)
        while (*body == ' ' || *body == '\t' || *body == '\n' || *body == '\r') body++;
        char *nl = strchr(body, '\n');
        if (nl) *nl = '\0';
        if (*body == '\0' && out[0] != '\0') {  // 공백·줄바꿈뿐이면 제안하지 않는다
            out[0] = '\0';
            *stop = "blank";
        }
    }
    return made;
}
