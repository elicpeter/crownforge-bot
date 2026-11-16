#!/usr/bin/env bash
# Builds what the arena simulator needs, in the active Python environment:
# the cr-rl-arena checkout at the pinned commit, this bot's bridge to it
# (cfbot_arena) and the stock simulator (cr_sim, for the parity tests).
#
#   python -m venv .venv && source .venv/bin/activate
#   pip install torch            # the build for your GPU, see the README
#   scripts/setup_arena.sh
#
# Needs git and a Rust toolchain (https://rustup.rs). ARENA_REV picks another
# commit: use the one the tournament server runs.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=${ARENA_REV:-21615a8275b1bad05ca4d7bc7ffb653e22bb8b8a}
if [ ! -d arena/cr-rl-arena/.git ]; then
  git clone https://github.com/ilumn/cr-rl-arena arena/cr-rl-arena
fi
git -C arena/cr-rl-arena fetch -q origin
git -C arena/cr-rl-arena checkout -q "$REV"
echo "cr-rl-arena at $(git -C arena/cr-rl-arena rev-parse --short HEAD)"
python -m pip install -q "maturin>=1.7,<2" "numpy>=1.26" "websockets>=14" pytest
(cd arena && maturin develop --release)
(cd arena/cr-rl-arena/gym && maturin develop --release)
# cfbot is plain Python on this simulator: if it is not installed, put python/ on the path.
if ! python -c "import cfbot" 2>/dev/null; then
  python - <<'PY'
import os, sysconfig
path = os.path.join(sysconfig.get_paths()["purelib"], "cfbot-src.pth")
open(path, "w").write(os.path.abspath("python") + "\n")
print("wrote", path)
PY
fi
python -c "import cfbot.arena, cfbot_arena, cr_sim, torch; print('arena bridge ready; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
