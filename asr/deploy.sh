#!/bin/sh
# Build only by default. Never creates, modifies, or deletes provider resources.
set -eu
compression=zstd
push=false
for argument in "$@"; do
  case "$argument" in
    --gzip) compression=gzip ;;
    --push) push=true ;;
    *) printf '%s\n' 'usage: sh asr/deploy.sh [--gzip] [--push]' >&2; exit 2 ;;
  esac
done
: "${IMAGE:?set IMAGE to the owner-approved registry tag}"
build_args=()
if [ "$(uname -m)" != x86_64 ]; then
  printf '%s\n' "Host architecture $(uname -m): skipping build-time NeMo connector assertion (runtime validation remains enabled)." >&2
  build_args+=(--build-arg ASR_VERIFY_NEMO_CONNECTOR=0)
fi
exec docker buildx build --platform linux/amd64 "${build_args[@]}" \
  -f asr/vast-serverless.Dockerfile \
  --output "type=image,name=$IMAGE,oci-mediatypes=true,compression=$compression,force-compression=true,push=$push" .
