import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path("/workspace")
models = root / "models"
snapshot_download("nvidia/parakeet-tdt-0.6b-v2", local_dir=models / "parakeet")
(root / "model-pins.json").write_text(json.dumps({
    "models": {"parakeet": "nvidia/parakeet-tdt-0.6b-v2"},
    "blocked_lanes": [],
}, indent=2) + "\n")
(root / ".models_ready").write_text("ready\n")
