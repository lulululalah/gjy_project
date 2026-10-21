# Compile away from open PDF readers, then publish the completed PDF atomically.
# Usage: powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File paper/build.ps1
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$manuscriptDirectory = Join-Path $PSScriptRoot 'manuscript'
$mainFile = Join-Path $manuscriptDirectory 'main.tex'
$buildDirectory = Join-Path $projectDirectory 'work/latex_manuscript_build'
$publishedPdf = Join-Path $manuscriptDirectory 'main.pdf'
$stagedPdf = Join-Path $manuscriptDirectory '.main.publish.pdf'
$latexmk = Get-Command latexmk.exe -CommandType Application -ErrorAction Stop |
    Select-Object -First 1

if (-not (Test-Path -LiteralPath $mainFile -PathType Leaf)) {
    throw "Manuscript source not found: $mainFile"
}
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
$buildLock = [System.IO.File]::Open(
    (Join-Path $buildDirectory '.build.lock'),
    [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None)
try {
    Push-Location -LiteralPath $manuscriptDirectory
    $previousTexInputs = $env:TEXINPUTS
    try {
        # LaTeX file-existence checks must also find the isolated auxiliary files.
        $env:TEXINPUTS = $buildDirectory.Replace('\', '/') + ';' + $previousTexInputs
        $buildArguments = @(
            '-pdf', '-interaction=nonstopmode', '-halt-on-error',
            '-file-line-error', '-synctex=1',
            ('-outdir=' + $buildDirectory.Replace('\', '/')), 'main.tex')
        & $latexmk.Source @buildArguments
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX build failed. See $(Join-Path $buildDirectory 'main.log')"
        }
    }
    finally {
        $env:TEXINPUTS = $previousTexInputs
        Pop-Location
    }

    $builtPdf = Join-Path $buildDirectory 'main.pdf'
    if (-not (Test-Path -LiteralPath $builtPdf -PathType Leaf)) {
        throw "LaTeX did not generate a PDF: $builtPdf"
    }
    if ((Test-Path -LiteralPath $publishedPdf -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $builtPdf).Hash -eq
         (Get-FileHash -LiteralPath $publishedPdf).Hash)) {
        Write-Host "PDF is up to date: $publishedPdf"
        return
    }

    # Rename the completed file; writing into a reader's mapped PDF can fail.
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ManuscriptPdfPublisher {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool MoveFileExW(string source, string target, uint flags);
}
'@
    Copy-Item -LiteralPath $builtPdf -Destination $stagedPdf -Force
    if (Test-Path -LiteralPath $publishedPdf -PathType Leaf) {
        $previousPdf = Join-Path $buildDirectory 'main.previous.pdf'
        Copy-Item -LiteralPath $publishedPdf -Destination $previousPdf -Force
    }
    # MOVEFILE_REPLACE_EXISTING; both paths are in the manuscript directory.
    if (-not [ManuscriptPdfPublisher]::MoveFileExW($stagedPdf, $publishedPdf, 1)) {
        $publishError = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw (New-Object System.ComponentModel.Win32Exception($publishError))
    }
    Write-Host "Published PDF: $publishedPdf"
}
finally {
    $buildLock.Dispose()
    if (Test-Path -LiteralPath $stagedPdf -PathType Leaf) {
        Remove-Item -LiteralPath $stagedPdf -Force
    }
}
