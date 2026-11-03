"""통합 테스트용 가짜 브리지: 보드 없이 bridge.py와 같은 JSON 한 줄 형식으로 답한다.
확장이 보내는 반응(shown, accept)은 FAKE_ACCEPT_LOG에 '<종류> <id>' 한 줄씩 남긴다."""
import json
import os
import sys

print(json.dumps({"ready": True, "port": "FAKE", "session": "test"}), flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") in ("accept", "shown"):
        with open(os.environ["FAKE_ACCEPT_LOG"], "a", encoding="utf-8") as f:
            f.write(f"{msg['type']} {msg['id']}\n")
        continue
    print(json.dumps({"id": msg["id"], "completion": "  # fake"}), flush=True)
