param(
    # A python.exe of exactly $pythonVersion, if the py launcher finds another 3.12.
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
# The trained Alien loads only on the exact Python it was trained on.
$pythonVersion = '3.12.5'
$pythonDownload = 'https://www.python.org/downloads/release/python-3125/'

function Invoke-Python([string[]]$command, [string[]]$arguments) {
    & $command[0] (@($command | Select-Object -Skip 1) + $arguments)
}

function Get-PythonVersion([string[]]$command) {
    try {
        return (Invoke-Python $command @('-c', 'import platform; print(platform.python_version())') 2>$null)
    } catch {
        return ''
    }
}

function Invoke-Bake([string]$output, [string]$module, [string]$without) {
    # A second try resumes, because each bake keeps the files it already downloaded.
    foreach ($try in 1..2) {
        if (Test-Path -LiteralPath (Join-Path $projectRoot $output)) { return }
        & $pythonPath -m $module
    }
    if ($LASTEXITCODE -ne 0) { Write-Warning "$module failed, so the game runs $without. Run this script again later." }
}

Push-Location -LiteralPath $projectRoot
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'Git is required, because the trained Alien checks the source it loads with: https://git-scm.com/download/win'
    }
    # A copy handed over without .git still needs a commit for that check to read.
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.git'))) {
        & git -c init.defaultBranch=main init -q
        & git -c user.name='Alien Survival' -c user.email='setup@localhost' commit -q --allow-empty -m 'Unpacked copy'
        if ($LASTEXITCODE -ne 0) { throw 'Could not start the git repository the checkpoint check reads.' }
    }
    if ((Test-Path -LiteralPath $pythonPath) -and (Get-PythonVersion @($pythonPath)) -ne $pythonVersion) {
        throw "The existing .venv is not Python $pythonVersion. Delete the .venv folder and run this script again."
    }
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        $base = if ($Python) { @($Python) } else { @('py', '-3.12') }
        $found = Get-PythonVersion $base
        if ($found -ne $pythonVersion) {
            throw ("Python $pythonVersion is required, found '$found'. Install it from $pythonDownload " +
                "and run this script again, or pass its python.exe with -Python <path>.")
        }
        Invoke-Python $base @('-m', 'venv', (Join-Path $projectRoot '.venv'))
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

    # The purchased textures never enter the repo: bake them from the pack, or use baked ones handed over privately.
    $texturePack = Join-Path $projectRoot 'assets\stradex\Materials'
    $bakedTextures = Join-Path $projectRoot 'assets\stradex\prepared\ship_textures.npz'
    if (Test-Path -LiteralPath $texturePack) {
        & $pythonPath -m tools.bake_ship_textures
        if ($LASTEXITCODE -ne 0) { throw 'Texture baking failed.' }
    }
    # Free sounds, the view from the bridge and the mood download once; the game runs without them.
    Invoke-Bake 'assets\sounds\prepared\ship_hum.wav' 'tools.bake_ship_sounds' 'silent'
    Invoke-Bake 'assets\space\ship_sky.npz' 'tools.bake_ship_sky' 'with dark windows'
    Invoke-Bake 'assets\mood\ship_mood.npz' 'tools.bake_ship_mood' 'with plain mist and no color grade'

    & $pythonPath -m tools.check_setup --rl-smoke
    if ($LASTEXITCODE -ne 0) { throw 'Setup verification failed.' }
    if (-not (Test-Path -LiteralPath $bakedTextures)) {
        Write-Warning ('The ship textures are missing, so the game cannot start yet. Put the licensed Stradex pack ' +
            'in assets\stradex\ and run this script again, or place baked textures in assets\stradex\prepared\.')
    }
} finally {
    Pop-Location
}
