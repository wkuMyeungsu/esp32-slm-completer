"""model.bin을 읽어 numpy로 한 토큰씩 추론한다 (torch 없이). 펌웨어 firmware/main/model.c가 이 구현을 그대로 옮겼다.

model.bin 형식 (리틀 엔디언):
  헤더 int32 8개: magic 0x314D4C53('SLM1'), dim, layers, heads, ffn, vocab, rot_dims, eos
  float32: embed_in[V,D], 층마다 ln1 w,b, ln2 w,b, qkv_w[3D,D], qkv_b, out_w[D,D], out_b,
           fc1_w[F,D], fc1_b, fc2_w[D,F], fc2_b, 그다음 final_ln w,b, embed_out[V,D]
  토큰 문자열: 토큰마다 uint8 길이 + UTF-8 바이트
  모델 버전: uint8 길이 + 문자열 (<체크포인트 이름>-<앞 내용 sha256 앞 8자리>)
"""
import math
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x314D4C53
MAX_NEW = 32
THETA, EPS = 10000.0, 1e-5


class NpModel:
    def __init__(self, path: Path):
        b = Path(path).read_bytes()
        magic, D, L, H, F, V, self.rot, self.eos = struct.unpack_from("<8i", b)
        assert magic == MAGIC, "model.bin 아님"
        self.D, self.L, self.H, self.hd = D, L, H, D // H
        off = 32

        def take(*shape):
            nonlocal off
            n = int(np.prod(shape))
            a = np.frombuffer(b, "<f4", n, off).reshape(shape)
            off += n * 4
            return a

        self.emb = take(V, D)
        self.layers = [[take(D), take(D), take(D), take(D), take(3 * D, D), take(3 * D), take(D, D), take(D),
                        take(F, D), take(F), take(D, F), take(D)] for _ in range(L)]
        self.lnf = (take(D), take(D))
        self.out = take(V, D)
        self.strs = []  # 토큰마다 바이트 (1바이트 토큰은 UTF-8 글자의 일부일 수 있어 문자열로 바꾸지 않는다)
        for _ in range(V):
            n = b[off]
            self.strs.append(b[off + 1:off + 1 + n])
            off += 1 + n
        # 끝: uint8 길이 + 모델 버전 (export.py가 씀, 예전 파일엔 없음)
        self.version = b[off + 1:off + 1 + b[off]].decode() if off < len(b) else "unknown"
        self.reset()

    def reset(self):
        self.pos = 0
        self.kc = [[] for _ in range(self.L)]
        self.vc = [[] for _ in range(self.L)]
        self.cached = []  # KV 캐시에 들어 있는 위치별 토큰

    def rewind(self, ids: list[int]) -> int:
        """펌웨어처럼 이전 계산과 겹치는 앞부분은 KV 캐시를 그대로 쓴다. 다시 계산할 시작 위치를 돌려준다."""
        n = 0
        while n < len(ids) - 1 and n < len(self.cached) and self.cached[n] == ids[n]:
            n += 1
        self.pos = n
        self.cached = self.cached[:n]
        self.kc = [k[:n] for k in self.kc]
        self.vc = [v[:n] for v in self.vc]
        return n

    def ln(self, x, w, b):
        m = x.mean()
        v = ((x - m) ** 2).mean()
        return (x - m) / math.sqrt(v + EPS) * w + b

    def rope(self, x):  # x [H, hd], 앞 rot 차원만 회전 (rotate_half 방식)
        half = self.rot // 2
        ang = self.pos * THETA ** (-np.arange(0, self.rot, 2, dtype=np.float64) / self.rot)
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        x = x.copy()
        x1, x2 = x[:, :half].copy(), x[:, half:self.rot].copy()
        x[:, :half] = x1 * c - x2 * s
        x[:, half:self.rot] = x2 * c + x1 * s
        return x

    def step(self, t: int) -> np.ndarray:
        self.cached.append(t)
        x = self.emb[t]
        for i, (l1w, l1b, l2w, l2b, qw, qb, ow, ob, f1w, f1b, f2w, f2b) in enumerate(self.layers):
            qkv = (qw @ self.ln(x, l1w, l1b) + qb).reshape(self.H, 3 * self.hd)  # 헤드마다 [q|k|v]
            q, k, v = (qkv[:, j * self.hd:(j + 1) * self.hd] for j in range(3))
            q, k = self.rope(q), self.rope(k)
            self.kc[i].append(k)
            self.vc[i].append(v)
            K, V = np.stack(self.kc[i], 1), np.stack(self.vc[i], 1)  # [H, T, hd]
            a = np.einsum("hd,htd->ht", q, K) / math.sqrt(self.hd)
            a = np.exp(a - a.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            att = np.einsum("ht,htd->hd", a, V).reshape(-1)
            m = f1w @ self.ln(x, l2w, l2b) + f1b
            m = 0.5 * m * (1 + np.vectorize(math.erf)(m / math.sqrt(2)))  # 정확한 GELU
            x = x + (ow @ att + ob) + (f2w @ m + f2b)  # parallel residual
        self.pos += 1
        return self.out @ self.ln(x, *self.lnf)

    def generate(self, ids: list[int], heal: str = "") -> str:
        """탐욕적 생성. heal이 있으면 생성 결과가 heal로 시작하도록 앞 토큰들을 제한하고, heal은 잘라서 돌려준다.
        글자가 나온 뒤 줄바꿈이 나오면 멈춘다 (완성은 한 줄)."""
        if not ids:
            return ""
        for t in ids[self.rewind(ids):-1]:
            self.step(t)
        heal_b = heal.encode()
        t, text = ids[-1], b""
        for _ in range(MAX_NEW):
            logits = self.step(t)
            if len(text) < len(heal_b):
                rest = heal_b[len(text):]
                ok = np.array([s != b"" and (s.startswith(rest) or rest.startswith(s)) for s in self.strs])
                logits = np.where(ok, logits, -np.inf)
            t = int(logits.argmax())
            if t == self.eos:
                break
            text += self.strs[t]
            body = text[len(heal_b):]
            if body.strip() and b"\n" in body.lstrip():
                break
        return text[len(heal_b):].decode(errors="replace") if text.startswith(heal_b) else ""


def encode(ids: list[int], decode, remap: dict) -> tuple[list[int], str]:
    """원래 토큰 id → (줄인 어휘 id, 힐링 문자열). bridge/bridge.py의 encode와 같은 규칙.
    토큰 힐링: 마지막 토큰을 떼고 생성 쪽에서 그 문자열을 다시 맞추게 한다. 원래 id에서 먼저 한다
    (끝 토큰이 줄인 어휘에 없는 경우가 많다, 예: 줄바꿈 3개짜리 토큰).
    shortcut: 줄인 어휘에 없는 나머지 토큰은 버린다, 학습 예시 밖 코드도 다루려면 바이트 토큰을 어휘에 넣고 다시 토큰화"""
    heal = ""
    if len(ids) > 1:
        ids, heal = ids[:-1], decode(ids[-1:])
    return [remap[t] for t in ids if t in remap], heal
