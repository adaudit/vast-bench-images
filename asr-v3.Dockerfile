# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755
ARG SOURCE_SHA=unresolved
LABEL org.opencontainers.image.source="https://github.com/adaudit/vast-bench-images" org.opencontainers.image.revision="${SOURCE_SHA}" org.opencontainers.image.licenses="CC-BY-4.0" io.adaudit.asr.schema="asr-candidate-v2" io.adaudit.asr.model="nvidia/parakeet-tdt-0.6b-v3" io.adaudit.asr.model-revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b" io.adaudit.asr.model-sha256="3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1
WORKDIR /workspace
COPY locks/parakeet-v3.requirements.txt /locks/parakeet-v3.requirements.txt
COPY locks/parakeet-v3.bootstrap.requirements.txt /locks/parakeet-v3.bootstrap.requirements.txt
COPY vendor/wheelhouses/parakeet-v3/antlr4_python3_runtime-4.9.3-py3-none-any.whl /bootstrap/
COPY vendor/wheelhouses/parakeet-v3/docopt-0.6.2-py2.py3-none-any.whl /bootstrap/
COPY vendor/wheelhouses/parakeet-v3/texterrors-0.4.4-cp311-cp311-linux_x86_64.whl /bootstrap/
COPY vendor/wheelhouses/parakeet-v3/wget-3.2-py3-none-any.whl /bootstrap/
RUN python3 -m pip uninstall -y ninja && python3 -m pip install --no-index --find-links /bootstrap --require-hashes -r /locks/parakeet-v3.bootstrap.requirements.txt && PIP_NO_INDEX=0 python3 -m pip install --require-hashes -r /locks/parakeet-v3.requirements.txt && python3 -m pip check
ADD --checksum=sha256:3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.nemo /workspace/models/parakeet-tdt-0.6b-v3.nemo
COPY asr/offline_entrypoint.py /workspace/offline_entrypoint.py
COPY asr/vast_adapter.py /workspace/vast_adapter.py
COPY asr/server.py /workspace/server.py
COPY asr/schemas/asr-candidate-v2.schema.json /workspace/asr-candidate-v2.schema.json
COPY asr/fixtures/parakeet-v3-calibration.jsonl /workspace/parakeet-v3-calibration.jsonl
RUN python3 -c "import hashlib, pathlib; p=pathlib.Path('/workspace/models/parakeet-tdt-0.6b-v3.nemo'); assert p.stat().st_size == 2509332480; assert hashlib.sha256(p.read_bytes()).hexdigest() == '3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'"
RUN mkdir /workspace/input /workspace/output && chown 65532:65532 /workspace/output
USER 65532:65532
ENTRYPOINT ["python3", "/workspace/server.py"]
