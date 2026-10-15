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

## Hourly global Vast ASR image

The production source is `asr/vast-serverless.Dockerfile`, **not** the old
backend `vast-serverless/` Test A v2 scaffold. The live model is
`nvidia/parakeet-tdt-0.6b-v3`, revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, original SHA256
`3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`.
NeMo is pinned to 2.4.1, torch to 2.5.1+cu124, and Vast SDK to 1.5.5.
The base Python 3.11 image remains digest-pinned. No model change is authorized.

`extract_model.py` checks the original archive, extracts once during build, and
bakes a per-file SHA256 manifest. Only the extracted tree enters the final image.
NeMo 2.4.1 `ModelPT.restore_from(..., save_restore_connector=connector)` uses
`connector.model_extracted_dir` for both its existence check and restore path;
`SaveRestoreConnector.load_config_and_state_dict` uses that directory instead
of unpacking a temporary archive. The image build asserts the connector property
exists. Each of three sequential load stages creates an independent model and
CUDA stream; a three-thread batch executor keeps the three lanes concurrent.

Startup stages: `deps`, `model_files`, `lane_1`, `lane_2`, `lane_3`,
`benchmark_1`, `benchmark_2`, `benchmark_3`, `ready`. Each benchmark targets
that exact model/stream; a retry cannot silently switch to another pool member.
Each successful stage writes an atomic `.done` audit marker under
`/tmp/asr-startup-markers` (override with `ASR_STARTUP_MARKER_DIR`). Failures retry
only the failed stage three times with 1s/2s backoff; completed models stay in
memory. Process restart must reload GPU objects: markers are not live CUDA state.
Each attempt emits content-free UTC `STAGE name start/end elapsed_ms=...` lines.
Only `ASR_STARTUP_FAILED` is registered with Vast `on_error`; per-item failures
do not kill the worker. There are no boot-time pip installs or model downloads.
The approved benchmark clip is baked locally. Startup deadline defaults to 1800s.

The backend UTC :00 timer invokes an authenticated global DB gate once hourly.
`ASR_BATCH_MIN_AUDIO_SECONDS` is a platform/server setting (default 10800).
No tenant scope, batch cap, max-wait fallback, endpoint override, or minute polling
is accepted on the Vast dispatch path. Existing per-tenant lease completion
authorization remains required. The gate advisory lock and durable job-event
hour ledger prevent duplicate dispatch after restart or concurrent processes.
The ledger is attached to an existing ASR job; deleting that anchor and all its
events during the same hour would erase that history. Apply backend migration
`migrations/add_asr_batch_event_indexes.sql` (rollback:
`migrations/rollback_asr_batch_event_indexes.sql`) for exact-lease event and
hour-ledger indexes. Local isolated PostgreSQL EXPLAIN tests verify both indexes.
The configured unbounded Vast identity cannot use legacy `/asr/claim`; other
identities and provider-transcript paths retain their separate contracts.
There is at most one run in flight; after it finishes, the first subsequent
UTC :00 boundary is evaluated normally, without catch-up or overlapping runs.

One route allocation (`cost=item count`; WAITING polls reuse `request_idx`)
and one physical worker POST carry
`{batch_id, items:[{id,audio_url,audio_refresh,lane,params}]}`.
`id` is a SHA256 of the tenant/job/lease/generation tuple, never a result index.
`params` carries immutable v3 model pins, duration, and the canonical WAV receipt.
Lease renewal, immutable-input preparation/dispatch fencing, and batch acceptance
are set-based control-plane operations, not N sequential HTTP requests.
`audio_refresh:{endpoint,token}` is a domain-separated HMAC capability scoped
to one exact tenant/job/worker/lease/generation and **only** audio re-signing.
Its lifetime follows that renewable DB lease; completion, expiry, reassignment,
or signing-key revocation invalidates it. It is not an ASR worker session.
When a download slot starts, it POSTs the capability to the fixed
`/internal/call-pipeline/asr/batch/audio-url` route. The controller rechecks the
persisted lease and issues a short signed URL capped by its actual expiry.
Production does not enqueue pre-signed URLs (`audio_url` is empty in the initial
payload); queued items cannot age a signature before starting their download.
No worker-session/bootstrap/provider credentials are sent to the GPU.

Eight concurrent downloads validate checksum/size/WAV metadata and reject private
addresses and redirects. A total 180-second download deadline covers bounded DNS,
connect, headers, and body reads; a socket-abort timer stops trickling peers.
The small JIT exchange has a separate 30-second deadline. Successful downloads
are submitted immediately to the three inference lanes, without waiting for
stalled peers. Per-item failures emit only id/status, never URLs or exceptions.

The response is `application/x-ndjson`: `accepted`, then `item` events
(`id,status=succeeded,candidate` or `id,status=failed`), optional `keepalive`,
then `end`, each tagged with `batch_id`. Go strictly validates each candidate and
publishes/completes it durably before reading the next item. Out-of-order items
map by id; duplicate/foreign ids are rejected. Explicit failures or missing ids
after a clean `end` requeue without attempt burn. Following acknowledged stream
interruption, the lease reaper requeues unfinished **unpublished** jobs without
an attempt increment. Before artifact publication, a durable per-item
`asr_publication_fenced` event takes precedence over batch acceptance; an upload
or completion that might have happened is never requeued automatically, even
if the failure-report RPC also fails. Physical worker POST attempts are tracked:
known preparation/fencing failures before that boundary release claims without
attempt burn; genuinely ambiguous POST/upload outcomes remain
`provider_post_uncertain`. Leases are not retired on failed failure transitions.

The installed `vastai-sdk==1.5.5` distribution is a compatibility wrapper pinning
`vastai==1.5.5`. Its worker backend `session` property uses
`aiohttp.ClientTimeout(total=None)` and `force_close=True`; `__handle_request`
calls `_request_start` before forwarding and `_request_end` only after the awaited
streaming response generator completes. Thus this implementation keeps **one
request in flight** for the whole batch, with 10s stream keepalives and no 220s
Go response deadline. There is no SDK hard limit requiring split POSTs.
Source: https://github.com/vast-ai/vast-cli/tree/1.5.5/vastai/serverless/server
(also inspect the exact PyPI 1.5.5 wheel). This verifies SDK behavior, not the
provider's external network/idle-shutdown service; that needs an owner-run canary.
On network partition an acknowledged unfinished item can be recomputed next
hour; full lease fencing prevents stale completion from overwriting the winner.

The pinned SDK's unused `/session/create`, `/session/end`, `/session/get`, and
`/session/health` handlers are disabled to return 404 **before** Worker starts,
including its internal HTTP session endpoint. This closes the unauthenticated
`on_close_route` callback/SSRF surface. `max_sessions=0` is not used (it means
unlimited in this SDK). SDK hardening fails closed if the real pinned SDK cannot
be patched; isolated contract-test SDK stubs are the only exception.

The image JSON schema mirrors and live Python candidate producer now enforce
the backend v2/v3 limit of 50000 segments. Previously the Go decoder enforced
50000, the unused image mirrors said 5000, and Python had no explicit count cap.
Legacy unversioned candidate decoding retains its separate 5000-segment contract.

Build from `vast-bench-images/` using `asr/deploy.sh`; it **never** calls the Vast
API or changes endpoint settings. The owner alone may publish the image and
update the existing template. Endpoint 35304 / workergroup 45437 remain min 0,
max 1, idle shutdown 60s; no code deletes instances or provisions warm workers.

```sh
IMAGE=ghcr.io/adaudit/vast-asr-hourly:v1 ./asr/deploy.sh
# Explicit owner-operated build AND push:
IMAGE=ghcr.io/adaudit/vast-asr-hourly:v1 ./asr/deploy.sh --push
# Registry/runtime that cannot consume zstd:
IMAGE=ghcr.io/adaudit/vast-asr-hourly:v1 ./asr/deploy.sh --gzip --push
```

The default BuildKit image output is
`type=image,oci-mediatypes=true,compression=zstd,force-compression=true,push=false`.
Local Docker `--load` is deliberately not used because it can discard exporter
compression; the gzip fallback uses the same image exporter. No paid run or
push is part of offline verification.
