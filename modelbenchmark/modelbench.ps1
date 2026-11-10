# Cross-platform launcher for Windows PowerShell.
#
#   .\modelbench.ps1 setup                      create .venv and install deps
#   .\modelbench.ps1 web [--port 8765]          local web console
#   .\modelbench.ps1 run --models minicpm5-1b   CLI benchmark
#
# Locates the repository from this script ($PSScriptRoot), never from the
# current directory.  Python discovery for setup: $env:MB_PYTHON -> py -3.12 ->
# python.  Other commands always run .venv\Scripts\python.exe; arguments are
# forwarded verbatim and the exit code is propagated.  This script has no side
# effects beyond running the requested command.
#
# NOTE: deliberately *no* param() block.  A param block with
# ValueFromRemainingArguments can interpret ``--version``/``--offline`` as
# PowerShell parameter names (and behaves differently between PS 5.1 and 7);
# ``$args`` receives every token exactly as written, including dash flags.
$ErrorActionPreference = "Continue"

function Get-PythonFromCommand {
    param(
        [string]$Executable,
        [string[]]$PrefixArgs = @()
    )
    try {
        $output = & $Executable @PrefixArgs -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) {
            return [string]($output | Select-Object -Last 1)
        }
    } catch {
        return $null
    }
    return $null
}

function Resolve-BasePython {
    if ($env:MB_PYTHON) {
        $resolved = Get-PythonFromCommand -Executable $env:MB_PYTHON
        if (-not $resolved) {
            [Console]::Error.WriteLine("modelbench: MB_PYTHON=$env:MB_PYTHON is not a runnable Python interpreter")
        }
        return $resolved
    }
    $resolved = Get-PythonFromCommand -Executable "py" -PrefixArgs @("-3.12")
    if (-not $resolved) {
        $resolved = Get-PythonFromCommand -Executable "python"
    }
    return $resolved
}

function Get-RemainingArgs {
    param([string[]]$Items)
    if ($null -eq $Items -or $Items.Count -le 1) {
        return @()
    }
    return $Items[1..($Items.Count - 1)]
}

$root = $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$exitCode = 0

Push-Location $root
try {
    $arguments = @($args)
    $command = ""
    if ($arguments.Count -gt 0) {
        $command = [string]$arguments[0]
    }

    if ($command -eq "setup") {
        $basePython = Resolve-BasePython
        if (-not $basePython) {
            [Console]::Error.WriteLine("modelbench: Python 3.10+ not found. Install Python (python.org) or set MB_PYTHON=C:\path\to\python.exe")
            $exitCode = 1
        } else {
            $bootstrap = Join-Path $root "tool\bootstrap.py"
            if (-not (Test-Path -LiteralPath $bootstrap)) {
                [Console]::Error.WriteLine("modelbench: missing $bootstrap")
                $exitCode = 1
            } else {
                $remainder = Get-RemainingArgs -Items $arguments
                & $basePython $bootstrap --venv (Join-Path $root ".venv") @remainder
                $exitCode = $LASTEXITCODE
            }
        }
    } elseif ($command -in @("", "help", "--help", "-h")) {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            [Console]::Error.WriteLine("environment not ready: run modelbench.ps1 setup")
            $exitCode = 2
        } else {
            & $venvPython -m modelbench.interfaces.cli --help
            $exitCode = $LASTEXITCODE
        }
    } else {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            [Console]::Error.WriteLine("environment not ready: run modelbench.ps1 setup")
            $exitCode = 2
        } else {
            & $venvPython -m modelbench.interfaces.cli @arguments
            $exitCode = $LASTEXITCODE
        }
    }
} finally {
    Pop-Location
}

exit $exitCode
