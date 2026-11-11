$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        & py -3.12 -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    $lockPath = Join-Path $projectRoot 'requirements.lock.txt'
    if (Test-Path -LiteralPath $lockPath) {
        & $pythonPath -m pip install -r $lockPath
        if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
    } else {
        & $pythonPath -m pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
        & $pythonPath -m pip install -r requirements.in
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    }
    & $pythonPath -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    & $pythonPath -m tools.bake_ship_textures
    if ($LASTEXITCODE -ne 0) { throw 'Texture baking failed.' }
    & $pythonPath -m tools.check_setup --rl-smoke
    if ($LASTEXITCODE -ne 0) { throw 'Setup verification failed.' }
} finally {
    Pop-Location
}
