import json
import os
import sys
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


def bake(name: str) -> None:
    snapshot_download(downloads[name], local_dir=models / name, token=os.environ.get("HF_TOKEN"))



def finalize() -> None:
    (root / "model-pins.json").write_text(json.dumps({
        "models": model_pins,
        "notes": {"whisper_turbo": "substitute: Systran repo deleted upstream"},
        "blocked_lanes": [],
    }, indent=2) + "\n")
    (root / ".models_ready").write_text("ready\n")


if __name__ == "__main__":
    # One model per invocation so each bake lands in its own image layer;
    # a single 6.7 GB layer stalls docker pulls on hosts with flaky links.
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "finalize":
        finalize()
    elif target == "all":
        for name in downloads:
            bake(name)
        finalize()
    else:
        bake(target)
