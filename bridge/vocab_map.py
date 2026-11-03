"""줄인 어휘로 토큰화한다. 어휘에 없는 토큰은 버리지 않고 어휘 안의 더 작은 토큰(최악은 1바이트 토큰)으로 쪼갠다.

학습 데이터 만들기(train/prep_data.py), 평가(train/eval_bin.py), 브리지가 모두 이 파일 하나를 쓴다.
세 곳의 규칙이 어긋나면 모델이 학습 때와 다른 토큰을 보게 되므로 복사하지 않는다.
의존성 없음 (토크나이저는 id → 토큰 문자열 함수만 받는다).
"""


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2·GPT-NeoX 바이트 단위 BPE가 바이트를 글자로 바꾸는 표 (토큰 문자열 → 원래 바이트를 되찾는 데 쓴다)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_U2B = {c: b for b, c in _bytes_to_unicode().items()}
NOT_IN_UTF8 = {0xC0, 0xC1, *range(0xF5, 0x100)}


def token_bytes(piece: str) -> bytes:
    """토큰 문자열(예: 'Ġprint') → 원래 바이트(b' print'). 특수 토큰(<|endoftext|>)은 글자 그대로."""
    if all(c in _U2B for c in piece):
        return bytes(_U2B[c] for c in piece)
    return piece.encode()


class Vocab:
    """keep: 남길 원래 토큰 id 목록(순서 = 줄인 id). id_to_token: 원래 id → 토큰 문자열."""

    def __init__(self, keep: list[int], id_to_token):
        self.keep = list(keep)
        self.id_to_token = id_to_token
        self.remap = {o: n for n, o in enumerate(self.keep)}
        self.bytes = [token_bytes(id_to_token(o)) for o in self.keep]
        self.by_bytes = {}
        for n, b in enumerate(self.bytes):
            self.by_bytes.setdefault(b, n)
        self.maxlen = max(map(len, self.bytes))
        # Pythia 토크나이저엔 1바이트 토큰이 243개뿐이다. 빠진 13개(0xC0, 0xC1, 0xF5~0xFF)는 UTF-8 문자열에 나올 수 없다
        missing = {b for b in range(256) if bytes([b]) not in self.by_bytes}
        assert missing <= NOT_IN_UTF8, f"UTF-8에 나오는 1바이트 토큰 {sorted(missing - NOT_IN_UTF8)}이 어휘에 없다"

    def split(self, b: bytes) -> list[int]:
        """바이트열을 어휘 안의 토큰으로 왼쪽부터 가장 긴 것부터 맞춰 자른다."""
        out, i = [], 0
        while i < len(b):
            for j in range(min(len(b), i + self.maxlen), i, -1):
                n = self.by_bytes.get(b[i:j])
                if n is not None:
                    out.append(n)
                    i = j
                    break
            else:
                i += 1  # UTF-8에 나올 수 없는 바이트 (어휘에 1바이트 토큰이 없음)
        return out

    def encode_prefix(self, ids: list[int]) -> tuple[list[int], str, int]:
        """앞부분의 원래 id → (줄인 id, 힐링 문자열, 쪼갠 토큰 수).
        토큰 힐링: 마지막 토큰을 떼고 그 문자열을 생성 쪽에서 다시 맞추게 한다 (앞부분이 공백으로 끝날 때 토큰 경계 문제).
        마지막 토큰이 UTF-8 글자의 일부라 문자열로 못 보내면 떼지 않는다."""
        heal = ""
        if len(ids) > 1:
            try:
                heal = token_bytes(self.id_to_token(ids[-1])).decode()
                ids = ids[:-1]
            except UnicodeDecodeError:
                pass
        out, fallback = self.encode_ids(ids)
        return out, heal, fallback

    def encode_ids(self, ids: list[int]) -> tuple[list[int], int]:
        """원래 id → (줄인 id, 쪼갠 토큰 수)."""
        out, fallback = [], 0
        for t in ids:
            n = self.remap.get(t)
            if n is not None:
                out.append(n)
            else:
                out += self.split(token_bytes(self.id_to_token(t)))
                fallback += 1
        return out, fallback


def _selftest() -> None:
    table = _bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert token_bytes("Ġprint") == b" print" and token_bytes("Ċ") == b"\n" and token_bytes("<|endoftext|>") == b"<|endoftext|>"
    pieces = {i: table[i] for i in range(256)}  # 가짜 토크나이저: id 0~255 = 바이트, 256 = ' print'
    pieces[256] = "Ġprint"
    pieces[257] = "Ġprintf"
    v = Vocab(list(range(257)), pieces.__getitem__)
    assert v.encode_ids([256]) == ([256], 0)
    assert v.encode_ids([257]) == ([256, ord("f")], 1)  # 어휘에 없는 ' printf' → ' print' + 'f'


_selftest()
