import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path("/workspace")
models = root / "models"
model_pins = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v2",
    "whisper_turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "whisper_v3": "openai/whisper-large-v3",
}
downloads = {
    "parakeet": model_pins["parakeet"],
    "whisper_turbo": model_pins["whisper_turbo"],
    "whisper_v3": "Systran/faster-whisper-large-v3",
}
for name, repo in downloads.items():
    snapshot_download(repo, local_dir=models / name, token=os.environ.get("HF_TOKEN"))
(root / "model-pins.json").write_text(json.dumps({
    "models": model_pins,
    "notes": {"whisper_turbo": "substitute: Systran repo deleted upstream"},
    "blocked_lanes": [],
}, indent=2) + "\n")
(root / ".models_ready").write_text("ready\n")
