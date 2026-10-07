#!/usr/bin/env bash
# Bootstrap the ARC environment on the compute box. No sudo required.
#
#   ssh prashant@100.103.26.120 'bash -s' < scripts/bootstrap_server.sh
#
# Idempotent: safe to re-run. Installs uv into ~/.local/bin, pins Python 3.12
# (system python is 3.14, which has no stable torch wheels), and builds the venv.

set -euo pipefail

ARC_ROOT="${ARC_ROOT:-$HOME/arc}"
PY_VERSION="3.12"

log() { printf '\n[bootstrap] %s\n' "$*"; }

# --- uv ---------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    log "uv present: $(uv --version)"
fi

# Make uv available in future non-interactive shells too.
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

# --- project dir ------------------------------------------------------------
mkdir -p "$ARC_ROOT"/{logs,results,data}
cd "$ARC_ROOT"

log "pinning Python $PY_VERSION"
uv python install "$PY_VERSION"

if [ ! -d .venv ]; then
    log "creating venv"
    uv venv --python "$PY_VERSION" .venv
fi

# --- torch (CUDA 12.6 wheels; driver 595.x supports it) ---------------------
log "installing torch + deps (this is the slow part)"
uv pip install --python .venv/bin/python \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision

uv pip install --python .venv/bin/python \
    numpy scipy scikit-learn pandas \
    ripser persim \
    hydra-core omegaconf \
    tqdm rich tensorboard pyyaml pillow

# --- verify -----------------------------------------------------------------
log "verifying"
.venv/bin/python - <<'PY'
import torch, sys
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__)
print("cuda avail ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     ", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram       {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB")
import ripser, sklearn, scipy
print("ripser     ", ripser.__version__ if hasattr(ripser, "__version__") else "ok")
PY

log "done. ARC_ROOT=$ARC_ROOT"
