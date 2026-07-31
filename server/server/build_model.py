import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / "model_parts"
OUTPUT = ROOT / "models" / "best.pt"

parts = sorted(PARTS.glob("part-*.txt"))
if not parts:
    raise SystemExit("YOLO 모델 조각을 찾을 수 없습니다.")

encoded = "".join(path.read_text(encoding="ascii") for path in parts)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_bytes(base64.b64decode(encoded))
print(f"Restored {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
