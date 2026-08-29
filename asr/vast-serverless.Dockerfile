# syntax=docker/dockerfile:1
FROM ghcr.io/adaudit/vast-bench-asr@sha256:85f43e0898a254bf56efdbe050167148dd8c910998c759669db9bc100dd5cd65

USER root
RUN PIP_NO_INDEX=0 python3 -m pip install --no-cache-dir vastai-sdk==1.5.5 && python3 -m pip check && python3 -I -c "import vastai"

COPY asr/vast-pyworker.py /workspace/vast-pyworker/worker.py
COPY asr/fixtures/audio/2086-149220-0033.wav /workspace/fixtures/2086-149220-0033.wav
COPY asr/vast-entrypoint.sh /workspace/vast-entrypoint.sh

RUN chmod +x /workspace/vast-entrypoint.sh

ENTRYPOINT ["/workspace/vast-entrypoint.sh"]
