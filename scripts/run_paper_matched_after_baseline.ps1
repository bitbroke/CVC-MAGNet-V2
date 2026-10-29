param(
    [Parameter(Mandatory = $true)]
    [int]$BaselinePid
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Ankshit\CVC-MAGNet-V2'
$queueDir = Join-Path $repo 'runs\paper_matched_single_cvc_v1'
$runDir = Join-Path $queueDir 'model_run'
$statusPath = Join-Path $queueDir 'queue_status.json'
$python = 'C:\Users\M S I\anaconda3\python.exe'
$stdout = Join-Path $queueDir 'training.log'
$stderr = Join-Path $queueDir 'training_stderr.log'

New-Item -ItemType Directory -Path $queueDir -Force | Out-Null
@{
    status = 'waiting_for_baseline'
    baseline_pid = $BaselinePid
    queued_at = (Get-Date).ToString('o')
    next_run = 'paper_matched_single_cvc_v1'
    output_dir = $runDir
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath

while ($true) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$BaselinePid" -ErrorAction SilentlyContinue
    if (-not $process -or $process.CommandLine -notlike '*train_placement_baseline.py*') {
        break
    }
    Start-Sleep -Seconds 15
}

if (Test-Path (Join-Path $runDir 'run_summary.json')) {
    throw "Refusing to overwrite an already completed run at $runDir"
}

@{
    status = 'starting'
    baseline_pid = $BaselinePid
    started_at = (Get-Date).ToString('o')
    output_dir = $runDir
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath

$arguments = @(
    '-u', 'scripts/train_single_cvc_placement.py',
    '--manifest', 'experiments/paper_matched_single_cvc_v1/single_annotated_cvc_manifest.parquet',
    '--output_dir', 'runs/paper_matched_single_cvc_v1/model_run',
    '--updates', '1000',
    '--eval_interval', '250',
    '--micro_batch', '8',
    '--effective_batch', '32',
    '--lr', '3e-4',
    '--seed', '42'
)
$training = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

@{
    status = 'running'
    baseline_pid = $BaselinePid
    training_pid = $training.Id
    started_at = (Get-Date).ToString('o')
    command = 'train_single_cvc_placement.py, 1000 updates, batch 32, seed 42'
    output_dir = $runDir
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath

$training.WaitForExit()
$exitCode = $training.ExitCode
@{
    status = if ($exitCode -eq 0) { 'complete' } else { 'failed' }
    baseline_pid = $BaselinePid
    training_pid = $training.Id
    exit_code = $exitCode
    finished_at = (Get-Date).ToString('o')
    output_dir = $runDir
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath

if ($exitCode -ne 0) {
    throw "Paper-matched single-CVC training exited with code $exitCode"
}
