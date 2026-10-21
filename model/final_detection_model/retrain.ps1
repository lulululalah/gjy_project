$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '4'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskOutput = Join-Path $taskRoot ('work/three_class_refinement_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& 'D:\Anaconda\envs\cad_graph_env\python.exe' -u (Join-Path $taskRoot 'python/main.py') train --output-dir $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Categorical refinement training or evaluation failed.' }
