param([switch]$InstallPyInstaller)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'

$tkReady = (& $python -c "import tkinter; ok=True;`ntry: tkinter.Tcl()`nexcept Exception: ok=False`nprint('yes' if ok else 'no')" 2>$null).Trim() -eq 'yes'
if (-not $tkReady) {
    Write-Warning 'Tkinter/Tcl no pudo validarse desde el entorno virtual. Se continuará: el EXE existente funciona en este equipo, pero repara Python con Tcl/Tk and IDLE si el nuevo EXE no abre la interfaz.'
}

$pyInstallerInstalled = (& $python -c "import importlib.util; print('yes' if importlib.util.find_spec('PyInstaller') else 'no')" 2>$null).Trim() -eq 'yes'
if (-not $pyInstallerInstalled) {
    if (-not $InstallPyInstaller) {
        throw 'PyInstaller no está instalado. Ejecuta: .\build_exe.ps1 -InstallPyInstaller'
    }
    & $python -m pip install --upgrade pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo instalar PyInstaller; revisa la conexión a PyPI e inténtalo de nuevo.'
    }
}

& $python -m PyInstaller --noconfirm --clean --onedir --windowed --name QwenSrtTts `
    --collect-all qwen_tts --collect-all transformers --collect-all librosa --exclude-module torchvision `
    --hidden-import tkinter --hidden-import tkinter.ttk `
    (Join-Path $root 'srt_tts_app.py')
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller no pudo crear la distribución. Cierra QwenSrtTts.exe y cualquier Explorador abierto en dist\QwenSrtTts antes de reintentar.'
}

 # PyInstaller collects MSVC runtime DLLs from scientific wheels into _internal.
 # Those copies conflict with c10.dll; the verified system VC++ runtime is used instead.
$internal = Join-Path $root 'dist\QwenSrtTts\_internal'
foreach ($runtimeDll in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') {
    $runtimePath = Join-Path $internal $runtimeDll
    if (Test-Path -LiteralPath $runtimePath) {
        Remove-Item -LiteralPath $runtimePath -Force
    }
}

$distribution = Join-Path $root 'dist\QwenSrtTts'
foreach ($folder in 'models', 'voices') {
    $target = Join-Path $root $folder
    $link = Join-Path $distribution $folder
    if (Test-Path -LiteralPath $target) {
        if (-not (Test-Path -LiteralPath $link)) {
            New-Item -ItemType Junction -Path $link -Target $target | Out-Null
        }
    }
    else {
        Write-Warning "No se creó el enlace $folder porque no existe $target"
    }
}

Write-Host 'EXE creado en dist\QwenSrtTts con enlaces a models\ y voices\.'
