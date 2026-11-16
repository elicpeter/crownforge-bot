# Builds what the arena simulator needs, in the active Python environment:
# the cr-rl-arena checkout at the pinned commit, this bot's bridge to it
# (cfbot_arena) and the stock simulator (cr_sim, for the parity tests).
#
#   python -m venv .venv; .\.venv\Scripts\Activate.ps1
#   pip install torch --index-url https://download.pytorch.org/whl/cu128   # see the README
#   powershell -ExecutionPolicy Bypass -File scripts\setup_arena.ps1
#
# Needs git, a Rust toolchain (https://rustup.rs) and the Visual Studio C++
# build tools. $env:ARENA_REV picks another commit: use the one the tournament
# server runs.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$rev = if ($env:ARENA_REV) { $env:ARENA_REV } else { "21615a8275b1bad05ca4d7bc7ffb653e22bb8b8a" }
if (-not (Test-Path "arena\cr-rl-arena\.git")) {
    git clone https://github.com/ilumn/cr-rl-arena arena\cr-rl-arena
    if ($LASTEXITCODE) { throw "git clone failed" }
}
git -C arena\cr-rl-arena fetch -q origin
git -C arena\cr-rl-arena checkout -q $rev
if ($LASTEXITCODE) { throw "git checkout failed" }
Write-Host "cr-rl-arena at $(git -C arena\cr-rl-arena rev-parse --short HEAD)"
python -m pip install -q "maturin>=1.7,<2" "numpy>=1.26" "websockets>=14" pytest
if ($LASTEXITCODE) { throw "pip install failed" }
Push-Location arena
maturin develop --release
if ($LASTEXITCODE) { Pop-Location; throw "building the bridge failed" }
Pop-Location
Push-Location arena\cr-rl-arena\gym
maturin develop --release
if ($LASTEXITCODE) { Pop-Location; throw "building cr_sim failed" }
Pop-Location
# cfbot is plain Python on this simulator: if it is not installed, put python\ on the path.
python -c "import cfbot" 2>$null
if ($LASTEXITCODE) {
    python -c "import os, sysconfig; p = os.path.join(sysconfig.get_paths()['purelib'], 'cfbot-src.pth'); open(p, 'w').write(os.path.abspath('python') + '\n'); print('wrote', p)"
}
python -c "import cfbot.arena, cfbot_arena, cr_sim, torch; print('arena bridge ready; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
