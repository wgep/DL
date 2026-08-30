# Run MotionBERT's infer_wild.py on all 9 videos in a loop.
# Run this INSIDE motionbert_env, from the MotionBERT repo folder.
#
# Usage:
#   .\run_all_motionbert.ps1

# change your path
$videosFolder = "path to your recorded videos folder"
# change your path
$jsonFolder   = "path to exported JSON files (from batch_export_halpe26.py)"
# change your path
$outputRoot   = "path to save MotionBERT outputs"

$videoFiles = @(
    "01_preparation",
    "02_grasp_birds_tail",
    "03_single_whip",
    "04_lift_hand",
    "05_white_crane",
    "06_brush_knee",
    "07_hold_lute",
    "08_pulling_blocking",
    "09_apparent_close",
    "10_cross_hands"
)

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

foreach ($name in $videoFiles) {
    $vidPath  = Join-Path $videosFolder "$name.mp4"
    $jsonPath = Join-Path $jsonFolder "${name}_halpe26.json"
    $outPath  = Join-Path $outputRoot $name

    if (-not (Test-Path $vidPath)) {
        Write-Host "SKIP (video not found): $name" -ForegroundColor Yellow
        continue
    }
    if (-not (Test-Path $jsonPath)) {
        Write-Host "SKIP (json not found): $name" -ForegroundColor Yellow
        continue
    }

    Write-Host "`n=== Processing: $name ===" -ForegroundColor Cyan
    python infer_wild.py --vid_path "$vidPath" --json_path "$jsonPath" --out_path "$outPath"
}

Write-Host "`nAll done. Outputs in: $outputRoot" -ForegroundColor Green
