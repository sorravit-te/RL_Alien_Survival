param(
    # A python.exe of exactly $pythonVersion; without one the script finds or downloads it.
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Windows PowerShell downloads crawl while drawing progress
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$setupDone = Join-Path $projectRoot '.venv\setup_complete.txt'
# The trained Alien loads only on the exact Python it was trained on.
$pythonVersion = '3.12.5'
# Machines without it, or without git, get project-local copies: official packages, checked by hash.
$portablePython = Join-Path $projectRoot '.python\tools\python.exe'
$pythonPackage = @{
    Url = 'https://api.nuget.org/v3-flatcontainer/python/3.12.5/python.3.12.5.nupkg'
    Sha256 = '5B0BEE0E786EFF3EEA0BCC3C05825E65658BBFC916B68731A962841CF3758FA5'
}
$portableGit = Join-Path $projectRoot '.git-portable\cmd'
$gitPackage = @{
    Url = 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip'
    Sha256 = '50B04B55425B5C465D076CDB184F63A0CD0F86F6EC8BB4D5860114A713D2C29A'
}

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

function Get-Verified([hashtable]$package, [string]$folder) {
    $download = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName() + '.zip')
    try {
        Write-Host "Downloading $($package.Url)"
        Invoke-WebRequest -Uri $package.Url -OutFile $download -UseBasicParsing
        if ((Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash -ne $package.Sha256) {
            throw "The download does not match its published hash: $($package.Url)"
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        if (Test-Path -LiteralPath $folder) { Remove-Item -LiteralPath $folder -Recurse -Force }
        [IO.Compression.ZipFile]::ExtractToDirectory($download, $folder)
    } finally {
        Remove-Item -LiteralPath $download -Force -ErrorAction SilentlyContinue
    }
}

function Install-CppRuntime {
    # PyTorch needs the Microsoft C++ runtime, which a fresh Windows may lack.
    if (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\msvcp140.dll')) { return }
    $installer = Join-Path ([IO.Path]::GetTempPath()) 'vc_redist.x64.exe'
    Write-Host 'Installing the Microsoft Visual C++ runtime; Windows asks for permission.'
    Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $installer -UseBasicParsing
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw 'The Visual C++ runtime installer is not signed by Microsoft.'
    }
    $process = Start-Process -FilePath $installer -ArgumentList '/install', '/passive', '/norestart' -Verb RunAs -Wait -PassThru
    if ($process.ExitCode -notin 0, 1638, 3010) { throw "The Visual C++ runtime did not install (code $($process.ExitCode))." }
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
        # The trained Alien checks the source it loads with git.
        if (-not (Test-Path -LiteralPath (Join-Path $portableGit 'git.exe'))) {
            Get-Verified $gitPackage (Join-Path $projectRoot '.git-portable')
        }
        $env:PATH = "$portableGit;$env:PATH"
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
        if ($Python) {
            $base = @($Python)
        } elseif ((Get-PythonVersion @('py', '-3.12')) -eq $pythonVersion) {
            $base = @('py', '-3.12')
        } else {
            if (-not (Test-Path -LiteralPath $portablePython)) {
                Get-Verified $pythonPackage (Join-Path $projectRoot '.python')
            }
            $base = @($portablePython)
        }
        $found = Get-PythonVersion $base
        if ($found -ne $pythonVersion) { throw "Python $pythonVersion is required, found '$found' at $($base -join ' ')." }
        Invoke-Python $base @('-m', 'venv', (Join-Path $projectRoot '.venv'))
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    Install-CppRuntime
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
    Set-Content -LiteralPath $setupDone -Value $pythonVersion -Encoding ascii
} finally {
    Pop-Location
}
