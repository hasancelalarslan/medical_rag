param(
  [string]$InFile    = "eval/model_outputs.csv",   # was $Input (conflicts with PS $input)
  [string]$OutDir    = "eval",
  [string]$BertModel = "roberta-large",
  [string]$PPLModel  = "gpt2",
  [string]$Lang      = "en",
  [int]$BatchSize    = 16,
  [switch]$Docker
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSCommandPath) | Out-Null
Set-Location -Path .. | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:TRANSFORMERS_NO_TORCHVISION = "1"

# Ensure NLTK corpora BEFORE run (METEOR needs wordnet on import)
$pythonPre = @"
import os, nltk
os.makedirs(os.path.expanduser('~/nltk_data'), exist_ok=True)
for p in ['punkt','wordnet','omw-1.4']:
    nltk.download(p, quiet=True)
print('[nltk] punkt/wordnet/omw-1.4 ready')
"@

if ($Docker) {
  docker compose exec api bash -lc "python -c \"$($pythonPre.Replace('`n',';').Replace('""','\"\"'))\""
} else {
  python -c $pythonPre
  if ($LASTEXITCODE -ne 0) { throw 'Failed to prepare NLTK data.' }
  python -m pip install --upgrade pip
  python -m pip install pandas numpy tqdm nltk bert-score rouge-score torch transformers regex
}

# Build run command
$cmd = @(
  "python","scripts/run_quality.py",
  "--input",$InFile,
  "--outdir",$OutDir,
  "--bert_model",$BertModel,
  "--ppl_model",$PPLModel,
  "--lang",$Lang,
  "--batch_size",$BatchSize
)

if ($Docker) {
  Write-Host "[run] docker compose exec api $($cmd -join ' ')"
  docker compose exec api bash -lc "$($cmd -join ' ')"
} else {
  Write-Host "[run] $($cmd -join ' ')"
  & $cmd
}

Write-Host "`n[done] Outputs in '$OutDir': quality_scores.csv, quality_report.md, quality_summary.json"
