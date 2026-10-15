"""Build-only extraction of the already pinned model; never downloads weights."""
import hashlib
import json
import os
import tarfile
from pathlib import Path

from offline_entrypoint import MODEL_PATH, verify_model


def extract():
    verify_model(MODEL_PATH)
    target = MODEL_PATH.parent / "parakeet-v3"
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(MODEL_PATH) as archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError("model archive contains non-regular entries")
        archive.extractall(target, filter="data")
    if not (target / "model_config.yaml").is_file() or not (target / "model_weights.ckpt").is_file():
        raise ValueError("model archive has an unsupported layout")
    hashes = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[str(path.relative_to(target))] = digest.hexdigest()
    (target / "sha256-manifest.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")
    # This build-time assertion imports torch through NeMo, which is unstable
    # under QEMU emulation. Runtime startup still validates model loading.
    if os.environ.get("ASR_VERIFY_NEMO_CONNECTOR", "1") != "0":
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
        connector = SaveRestoreConnector()
        connector.model_extracted_dir = str(target)
        assert connector.model_extracted_dir == str(target)


if __name__ == "__main__":
    extract()
