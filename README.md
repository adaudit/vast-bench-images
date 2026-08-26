# Vast benchmark images

CUDA 12.x Vast Serverless benchmark images with their model weights baked into
the image layers. Build and publish with the `Build and publish benchmark images`
workflow. The diarization build requires the repository `HF_TOKEN` Actions secret.

## ASR-v3 runtime boundary

Run the candidate ASR-v3 image with a read-only root filesystem and model layer,
a read-only mount at `/workspace/input`, and a writable `tmpfs` at
`/workspace/output`. Use numeric UID/GID `65532:65532`, drop every Linux
capability, and set `no-new-privileges`; do not mount a replacement model or
enable network access. The entrypoint accepts only regular, non-symlink audio
files below `/workspace/input` and writes only below `/workspace/output`.
