import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path("/workspace")
models = root / "models"
model_pins = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v2",
    "whisper_turbo": "openai/whisper-large-v3-turbo",
    "whisper_v3": "openai/whisper-large-v3",
}
downloads = {
    "parakeet": model_pins["parakeet"],
    "whisper_turbo": "Systran/faster-whisper-large-v3-turbo",
    "whisper_v3": "Systran/faster-whisper-large-v3",
}
for name, repo in downloads.items():
    snapshot_download(repo, local_dir=models / name)
(root / "model-pins.json").write_text(json.dumps({
    "models": model_pins,
    "blocked_lanes": [],
}, indent=2) + "\n")
(root / ".models_ready").write_text("ready\n")
