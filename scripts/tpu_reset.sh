#!/bin/bash
# Surgical TPU reset for sglang iteration loop.
# Targets: lingering spawn children, vfio fd leaks, libtpu lockfile, runtime state.
# Exits 0 only if jax.devices() works afterwards.

# Neither -e nor -u — each cleanup step must run even if earlier ones miss.

echo "[tpu_reset] $(date '+%H:%M:%S') start"

# 1. Kill anything sglang-related, including reparented spawn children.
#    pkill matches against argv, so multiprocessing children running
#    `from multiprocessing.spawn import spawn_main; spawn_main(...)`
#    are caught by the spawn pattern.
#    Patterns are narrowed to avoid matching this script (its path
#    contains "sglang") or the current shell.
pkill -9 -f 'sglang.launch_server' 2>/dev/null
pkill -9 -f 'sglang/launch_server' 2>/dev/null
pkill -9 -f 'multiprocessing.spawn' 2>/dev/null
pkill -9 -f 'spawn_main' 2>/dev/null
pkill -9 -f 'run_scheduler_process' 2>/dev/null
pkill -9 -f 'run_tokenizer_process' 2>/dev/null
pkill -9 -f 'run_detokenizer' 2>/dev/null
pkill -9 -f 'resource_tracker' 2>/dev/null
echo "[tpu_reset] pkill round 1 done"

# Wait for SIGKILL to propagate.
sleep 2

# 2. Anyone holding /dev/vfio/*? Force-close.
#    fuser -k sends SIGKILL to processes that have any of those open.
for d in /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 \
         /dev/vfio/4 /dev/vfio/5 /dev/vfio/6 /dev/vfio/7; do
    sudo fuser -k -9 "$d" 2>/dev/null
done
echo "[tpu_reset] fuser -k done"

sleep 1

# 3. Remove stale lockfile.
sudo rm -f /tmp/libtpu_lockfile
echo "[tpu_reset] lockfile cleared"

# 4. Restart TPU runtime — drains residual driver state.
sudo systemctl restart tpu-runtime
echo "[tpu_reset] tpu-runtime restarted"

# 5. Give the runtime time to come up. tpu-runtime can take 15-20s on
#    this host before libtpu is usable; a too-short wait makes the
#    verify step race into "ABORTED: libtpu lockfile" or similar.
sleep 15

# 6. Verify we can claim TPU again. Retry up to 3 times with backoff
#    because the runtime is sometimes slow to settle.
cd ~/cuiq-sglang
# shellcheck disable=SC1091
source ~/sglang_tpu_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate

VERIFY_OK=0
for attempt in 1 2 3; do
    if python -c "
import jax, sys
try:
    ds = jax.devices()
except Exception as e:
    print(f'[tpu_reset] jax.devices() raised {type(e).__name__}: {str(e)[:150]}')
    sys.exit(1)
if len(ds) != 8:
    print(f'[tpu_reset] jax.devices() -> {len(ds)} chips (expected 8)')
    sys.exit(1)
print(f'[tpu_reset] jax.devices() -> {len(ds)} chips')
" 2>&1; then
        VERIFY_OK=1
        break
    else
        echo "[tpu_reset] verify attempt $attempt failed; cleaning + retrying"
        sudo rm -f /tmp/libtpu_lockfile
        sleep 5
    fi
done

if [ "$VERIFY_OK" = "1" ]; then
    echo "[tpu_reset] verify python exited; waiting for kernel to release vfio"
    # The verify python process holds /dev/vfio/* via libtpu while it
    # runs. Even after it exits cleanly, the kernel needs MANY seconds
    # to decrement refcounts and release the iommu group. Without enough
    # wait, the next process (e.g. sglang launch_server's worker) races
    # into "Device or resource busy" from /dev/vfio/N. Empirically 10 s
    # is required on this host even though `lsof` shows no holders.
    sleep 10
    # Wipe the lockfile the verify process left behind.
    sudo rm -f /tmp/libtpu_lockfile
    echo "[tpu_reset] $(date '+%H:%M:%S') OK"
    exit 0
else
    echo "[tpu_reset] $(date '+%H:%M:%S') FAIL — jax.devices() did not return 8 chips after 3 attempts"
    exit 1
fi
