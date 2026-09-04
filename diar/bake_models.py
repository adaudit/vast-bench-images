"""Build-time assertion for the model fetched and verified by the workflow."""
import hashlib
import sys
from pathlib import Path

MODEL_SIZE_BYTES = 471367680
MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"


def verify(path):
    if path.stat().st_size != MODEL_SIZE_BYTES:
        raise SystemExit("baked Sortformer model size does not match")
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise SystemExit("baked Sortformer model digest does not match")


if __name__ == "__main__":
    verify(Path(sys.argv[1] if len(sys.argv) == 2 else "/workspace/models/diar_streaming_sortformer_4spk-v2.1.nemo"))
