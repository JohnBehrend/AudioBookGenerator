#!/usr/bin/env bash
# Run the real-engine integration tests sequentially, one engine at a time.
#
# The real-engine tests load large TTS models (dramabox, omni, zonos2) that
# together exceed the 24GB VRAM of a single GPU, so they cannot all be loaded
# at once in one process. Running each engine in its own pytest process keeps
# them sequential (never parallel) and frees GPU memory when each process exits.
#
# Usage:
#   scripts/run-real-engine-tests.sh                 # default suite + every engine
#   scripts/run-real-engine-tests.sh omni zonos2     # only these engines
#   scripts/run-real-engine-tests.sh --skip-default  # skip the non-GPU suite
set -euo pipefail

cd "$(dirname "$0")/.."

SKIP_DEFAULT=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" == "--skip-default" ]; then
    SKIP_DEFAULT=1
  else
    ARGS+=("$arg")
  fi
done

ENGINES=()
if [ "${#ARGS[@]}" -gt 0 ]; then
  ENGINES=("${ARGS[@]}")
else
  mapfile -t ENGINES < <(uv run python -c "from tts import list_engines; print('\\n'.join(list_engines()))")
fi

if [ "$SKIP_DEFAULT" -eq 0 ]; then
  echo "=== Default suite (no GPU) ==="
  uv run pytest tests/ -q -p no:cacheprovider
fi

for engine in "${ENGINES[@]}"; do
  echo
  echo "=== Real-engine tests: $engine ==="
  # A fresh pytest process per engine: sequential execution, and the model's
  # VRAM is released when this process exits (before the next engine starts).
  uv run pytest tests/test_real_engines.py --run-slow --run-generate \
    -q -p no:cacheprovider -k "$engine"
done

echo
echo "All engines passed."
