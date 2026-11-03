"""평가를 GPU로 빠르게: 학습 체크포인트(어휘를 줄인 모델)를 torch로 돌리되, 생성 규칙은 npmodel.NpModel.generate를
그대로 쓴다 (힐링·반복 멈춤·줄 끝 멈춤이 보드와 같은 코드). NpModel과 같은 모양(reset, rewind, step, strs, eos, version)이다.

보드와 같은 계산인지는 export.py가 model.bin을 numpy로 읽어 torch와 비교해 확인한다 (로짓 최대 오차 1e-3 미만).
여기서는 그 torch 쪽을 GPU에서 돌려 시간만 줄인다. CI(GPU 없음)는 계속 numpy(model.bin)로 잰다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "bridge"))

from npmodel import NpModel  # noqa: E402
from vocab_map import token_bytes  # noqa: E402


class TorchModel:
    generate = NpModel.generate  # 생성 규칙은 numpy 모델과 같은 함수

    def __init__(self, ckpt: Path, device: str | None = None):
        ckpt = Path(ckpt)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32).to(self.device).eval()
        tok = AutoTokenizer.from_pretrained(ckpt)
        keep = json.loads((ckpt / "keep.json").read_text(encoding="utf-8"))["keep"]
        self.strs = [token_bytes(tok.convert_ids_to_tokens(o)) for o in keep]
        self.eos = keep.index(tok.eos_token_id)
        self.version = f"{ckpt.name} (torch, {self.device})"
        self.reset()

    def reset(self):
        self.past, self.cached, self.pos = None, [], 0

    def rewind(self, ids: list[int]) -> int:
        """이전 계산과 겹치는 앞부분은 KV 캐시를 그대로 쓴다 (NpModel.rewind와 같은 규칙)."""
        n = 0
        while n < len(ids) - 1 and n < len(self.cached) and self.cached[n] == ids[n]:
            n += 1
        if self.past is not None and n < len(self.cached):
            self.past.crop(n - len(self.cached))  # transformers 5: 뒤에서 지울 개수를 음수로 준다
        self.cached, self.pos = self.cached[:n], n
        return n

    @torch.no_grad()
    def feed(self, tokens: list[int]) -> None:
        """앞부분 토큰을 한 번의 계산으로 KV 캐시에 넣는다 (토큰마다 GPU를 부르면 호출 비용이 더 크다)."""
        if not tokens:
            return
        out = self.model(input_ids=torch.tensor([tokens], device=self.device), past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.cached += list(tokens)
        self.pos += len(tokens)

    @torch.no_grad()
    def step(self, t: int) -> np.ndarray:
        out = self.model(input_ids=torch.tensor([[t]], device=self.device), past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.cached.append(t)
        self.pos += 1
        return out.logits[0, -1].float().cpu().numpy()
