param()

$ErrorActionPreference = 'Stop'
$repo = 'C:\Ankshit\CVC-MAGNet-V2'
$python = 'C:\Users\M S I\anaconda3\python.exe'
$queuedStatus = Join-Path $repo 'runs\paper_matched_single_cvc_v1\queue_status.json'
$queueRoot = Join-Path $repo 'runs\placement_ablation_queue_v1'
New-Item -ItemType Directory -Path $queueRoot -Force | Out-Null
$topStatus = Join-Path $queueRoot 'queue_status.json'

function Write-Status([string]$status, [string]$stage, [hashtable]$extra = @{}) {
    $obj = @{ status = $status; stage = $stage; updated_at = (Get-Date).ToString('o') }
    foreach ($key in $extra.Keys) { $obj[$key] = $extra[$key] }
    $obj | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $topStatus
}

Write-Status 'waiting' 'paper_matched_single_cvc_v1' @{}
while ($true) {
    if (-not (Test-Path $queuedStatus)) { Start-Sleep -Seconds 15; continue }
    $state = Get-Content -LiteralPath $queuedStatus -Raw | ConvertFrom-Json
    if ($state.status -eq 'complete') { break }
    if ($state.status -eq 'failed') { Write-Status 'blocked' 'paper_matched_single_cvc_v1' @{ reason = 'upstream single-CVC pilot failed' }; exit 2 }
    Start-Sleep -Seconds 15
}

$jobs = @(
    @{
        name = 'all_cvc_traced_effnet_multilabel_1000'
        manifest = 'experiments/placement_cohort_matrix_v1/all_cvc_traced_multilabel.parquet'
        backbone = 'tf_efficientnet_b0.ns_jft_in1k'
    },
    @{
        name = 'full_cohort_resnet18_multilabel_1000'
        manifest = 'data/splits/placement_locked_manifest.parquet'
        backbone = 'resnet18.a1_in1k'
    }
)

foreach ($job in $jobs) {
    $runDir = Join-Path $repo ("runs\placement_ablation_queue_v1\" + $job.name)
    if (Test-Path $runDir) {
        $items = Get-ChildItem -LiteralPath $runDir -Force -ErrorAction SilentlyContinue
        if ($items.Count -gt 0) { Write-Status 'blocked' $job.name @{ reason = "refusing to overwrite $runDir" }; exit 3 }
    }
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    Write-Status 'running' $job.name @{ output_dir = $runDir; manifest = $job.manifest; backbone = $job.backbone }
    $outLog = Join-Path $runDir 'training.log'
    $errLog = Join-Path $runDir 'training_stderr.log'
    $args = @(
        '-u', 'scripts/train_placement_baseline.py',
        '--output_dir', $runDir,
        '--manifest', $job.manifest,
        '--backbone', $job.backbone,
        '--updates', '1000',
        '--eval_interval', '250',
        '--micro_batch', '8',
        '--effective_batch', '32',
        '--lr', '3e-4'
    )
    $proc = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
    Write-Status 'running' $job.name @{ training_pid = $proc.Id; output_dir = $runDir; manifest = $job.manifest; backbone = $job.backbone }
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
        Write-Status 'failed' $job.name @{ exit_code = $proc.ExitCode; output_dir = $runDir; log = $outLog }
        exit $proc.ExitCode
    }
    $summary = Join-Path $runDir 'placement_baseline_summary.json'
    if (-not (Test-Path $summary)) { Write-Status 'failed' $job.name @{ reason = 'summary missing'; output_dir = $runDir }; exit 4 }
    Write-Status 'complete' $job.name @{ output_dir = $runDir; summary = $summary }
}

$evalDir = Join-Path $queueRoot 'common_validation_evaluation'
New-Item -ItemType Directory -Path $evalDir -Force | Out-Null
Write-Status 'running' 'common_validation_evaluation' @{ output_dir = $evalDir }
$evalArgs = @(
    '-u', 'scripts/evaluate_placement_checkpoint_matrix.py',
    '--checkpoint', 'full_effnet|tf_efficientnet_b0.ns_jft_in1k|runs/placement_image_baseline_512/checkpoints/best_placement_baseline.pt',
    '--checkpoint', 'traced_effnet|tf_efficientnet_b0.ns_jft_in1k|runs/placement_ablation_queue_v1/all_cvc_traced_effnet_multilabel_1000/checkpoints/best_placement_baseline.pt',
    '--checkpoint', 'full_resnet18|resnet18.a1_in1k|runs/placement_ablation_queue_v1/full_cohort_resnet18_multilabel_1000/checkpoints/best_placement_baseline.pt',
    '--output', 'runs/placement_ablation_queue_v1/common_validation_evaluation/common_validation_metrics.json'
)
$evalProc = Start-Process -FilePath $python -ArgumentList $evalArgs -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $evalDir 'evaluation.log') `
    -RedirectStandardError (Join-Path $evalDir 'evaluation_stderr.log') -WindowStyle Hidden -PassThru
$evalProc.WaitForExit()
if ($evalProc.ExitCode -ne 0) {
    Write-Status 'failed' 'common_validation_evaluation' @{ exit_code = $evalProc.ExitCode; output_dir = $evalDir }
    exit $evalProc.ExitCode
}

Write-Status 'complete' 'all_jobs' @{ completed_runs = @($jobs | ForEach-Object { $_.name }) }
