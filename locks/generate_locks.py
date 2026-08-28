#!/usr/bin/env python3
"""Turn pip's target-runtime reports into offline, hash-locked inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

TARGET = "linux/amd64; CPython 3.11; pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
ANTLR = "antlr4-python3-runtime"
PARAKEET_BOOTSTRAPS = {
    "antlr4_python3_runtime-4.9.3-py3-none-any.whl": {
        "source_archive": "https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz",
        "source_archive_sha256": "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
    },
    "docopt-0.6.2-py2.py3-none-any.whl": {
        "source_archive": "local audited source archive: docopt-0.6.2.tar.gz",
        "source_archive_sha256": "49b3a825280bd66b3aa83585ef59c4a8c82f2c8a522dbe754a8bc8d08c85c491",
    },
    "texterrors-0.4.4-cp311-cp311-linux_x86_64.whl": {
        "source_archive": "local audited source archive: texterrors-0.4.4.tar.gz",
        "source_archive_sha256": "899892fca086939b22513ee33cb28ef191490f4a32e042e7abe97206c7769f4b",
    },
    "wget-3.2-py3-none-any.whl": {
        "source_archive": "https://files.pythonhosted.org/packages/47/6a/62e288da7bcda82b935ff0c6cfe542970f04e29c756b0e147251b2fb251f/wget-3.2.zip",
        "source_archive_sha256": "35e630eca2aa50ce998b9b1a127bb26b30dfee573702782aa982f875e3f16061",
    },
}


def artifact(item, antlr_wheel):
    info = item["download_info"]
    metadata = item["metadata"]
    url = info["url"]
    filename = Path(urlparse(url).path).name
    digest = info["archive_info"]["hashes"].get("sha256")
    if not digest or not url.startswith("https://files.pythonhosted.org/"):
        raise ValueError(f"untrusted or unhashed artifact: {metadata['name']}")
    if filename.endswith(".whl"):
        return {"name": metadata["name"], "version": metadata["version"], "filename": filename,
                "source": url, "sha256": digest, "compatibility": TARGET}
    if metadata["name"].lower() != ANTLR or filename != "antlr4-python3-runtime-4.9.3.tar.gz":
        raise ValueError(f"source distribution is forbidden: {filename}")
    if hashlib.sha256(antlr_wheel.read_bytes()).hexdigest() != "ffb51877578142e5df83abad6f59eed91050c79e4b4fc511dc12f3effddaddc6":
        raise ValueError("approved ANTLR wheel hash mismatch")
    return {"name": metadata["name"], "version": metadata["version"], "filename": antlr_wheel.name,
            "source": "local audited bootstrap", "source_archive": url, "source_archive_sha256": digest,
            "sha256": hashlib.sha256(antlr_wheel.read_bytes()).hexdigest(), "compatibility": TARGET}


def write_lock(report_path, stem, out, antlr_wheel):
    report = json.loads(report_path.read_text())
    artifacts = sorted((artifact(item, antlr_wheel) for item in report["install"]), key=lambda x: x["name"].lower())
    if len({a["name"].lower() for a in artifacts}) != len(artifacts):
        raise ValueError(f"duplicate package in {stem}")
    requirements = "\n".join(f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}" for a in artifacts) + "\n"
    (out / f"{stem}.requirements.txt").write_text(requirements)
    remote = [a for a in artifacts if a["source"] != "local audited bootstrap"]
    (out / f"{stem}.remote.requirements.txt").write_text(
        "\n".join(f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}" for a in remote) + "\n"
    )
    (out / f"{stem}.artifacts.json").write_text(json.dumps({"target": TARGET, "artifacts": artifacts}, indent=2) + "\n")


def write_parakeet_lock(report_path, out, local_wheels, append=False, replace=False):
    report = json.loads(report_path.read_text())
    manifest_path = out / "parakeet-v3.artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = list(manifest["wheels"]) if append else []
    names = {artifact["name"].lower() for artifact in artifacts}
    for item in report["install"]:
        info, metadata = item["download_info"], item["metadata"]
        filename = Path(urlparse(info["url"]).path).name
        digest = info["archive_info"]["hashes"].get("sha256")
        if not digest:
            raise ValueError(f"unhashed artifact: {metadata['name']}")
        if info["url"].startswith("https://files.pythonhosted.org/") and filename.endswith(".whl"):
            source = {"source": info["url"]}
        elif filename in PARAKEET_BOOTSTRAPS and filename in local_wheels:
            wheel = local_wheels[filename]
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != digest:
                raise ValueError(f"local bootstrap hash mismatch: {filename}")
            source = {"source": "local audited bootstrap", **PARAKEET_BOOTSTRAPS[filename]}
        else:
            raise ValueError(f"forbidden non-binary or unapproved local artifact: {filename}")
        if metadata["name"].lower() in names:
            if not replace:
                raise ValueError(f"duplicate package in parakeet-v3: {metadata['name']}")
            artifacts[:] = [artifact for artifact in artifacts if artifact["name"].lower() != metadata["name"].lower()]
        artifacts.append({"name": metadata["name"], "version": metadata["version"], "filename": filename,
                          "sha256": digest, "compatibility": TARGET, **source})
        names.add(metadata["name"].lower())
    artifacts.sort(key=lambda x: x["name"].lower())
    if len({a["name"].lower() for a in artifacts}) != len(artifacts):
        raise ValueError("duplicate package in parakeet-v3")
    (out / "parakeet-v3.requirements.txt").write_text(
        "\n".join(f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}" for a in artifacts) + "\n")
    manifest.update({"resolution_required": False, "wheels": artifacts})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--qwen-report", type=Path)
parser.add_argument("--pyannote-report", type=Path)
parser.add_argument("--antlr-wheel", type=Path)
parser.add_argument("--parakeet-report", type=Path)
parser.add_argument("--parakeet-local-wheel", action="append", default=[], metavar="PATH")
parser.add_argument("--parakeet-append-report", action="store_true")
parser.add_argument("--parakeet-replace-report", action="store_true")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
if args.parakeet_report:
    local_wheels = {path.name: path for path in map(Path, args.parakeet_local_wheel)}
    write_parakeet_lock(args.parakeet_report, args.out, local_wheels, args.parakeet_append_report,
                        args.parakeet_replace_report)
else:
    if not all((args.qwen_report, args.pyannote_report, args.antlr_wheel)):
        parser.error("legacy generation requires both reports and --antlr-wheel")
    write_lock(args.qwen_report, "asr", args.out, args.antlr_wheel)
    write_lock(args.pyannote_report, "diar", args.out, args.antlr_wheel)
