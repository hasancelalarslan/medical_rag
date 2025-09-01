# tests\test_medical_rag.ps1
# PowerShell 5.1 compatible.

$ErrorActionPreference = 'Stop'

# -------- Config --------
$ApiUrl    = $env:API_URL;    if ([string]::IsNullOrWhiteSpace($ApiUrl))    { $ApiUrl    = "http://127.0.0.1:8000/query" }
$HealthUrl = $env:HEALTH_URL; if ([string]::IsNullOrWhiteSpace($HealthUrl)) { $HealthUrl = "http://127.0.0.1:8000/health" }
$Root      = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputCsv  = Join-Path $Root "examples\queries_100.csv"
$OutDir    = Join-Path $Root "results"
$OutCsv    = Join-Path $OutDir "perf_results.csv"
$ComposeLogsPath = Join-Path $OutDir "compose_logs.txt"
$RunLog    = Join-Path $OutDir "log.txt"

$HealthMaxRetries = 60   # ~3 minutes total (60 * 3s)
$HealthSleepSec   = 3
$Services = @('api','ui') # which services to start

# -------- Helpers --------
function Log([string]$Msg) {
  $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line  = "[$stamp] $Msg"
  Write-Host $line
  Add-Content -Path $RunLog -Value $line
}

# Pick compose CLI ('docker compose' vs 'docker-compose')
$ComposeCmd = @('docker','compose')
$ComposeOK  = $false
try { & docker compose version *> $null; if ($LASTEXITCODE -eq 0) { $ComposeOK = $true } } catch {}
if (-not $ComposeOK) {
  try { & docker-compose version *> $null; if ($LASTEXITCODE -eq 0) { $ComposeCmd = @('docker-compose'); $ComposeOK = $true } } catch {}
}
if (-not $ComposeOK) { throw "Neither 'docker compose' nor 'docker-compose' is available on PATH." }

# Run compose with args array
function Compose([string[]]$Args) { & $ComposeCmd @Args }

# Ensure output dir & reset logs
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
"" | Out-File -FilePath $RunLog -Encoding UTF8

# Always down the stack on exit
$global:__stackUp = $false
$cleanup = {
  if ($global:__stackUp) {
    try {
      Log "Stopping docker compose..."
      Compose @('down') | Out-Null
      Log "Stack stopped."
    } catch { Write-Warning $_ }
  }
}
Register-EngineEvent PowerShell.Exiting -Action $cleanup | Out-Null

# -------- Bring up --------
Log "Starting docker compose (detached, build if needed)..."
Compose @('up','-d','--build') + $Services | Out-Null
$global:__stackUp = $true

# -------- Wait for health --------
Log "Waiting for health at $HealthUrl ..."
$healthy = $false
$startupErr = $null
for ($i=0; $i -lt $HealthMaxRetries; $i++) {
  try {
    $health = Invoke-RestMethod $HealthUrl -TimeoutSec 3
    $status = [string]$health.status
    if ($health.PSObject.Properties.Name -contains 'startup_error') {
      $startupErr = [string]$health.startup_error
    }
    if ($status -eq 'ok') { $healthy = $true; break }
    # Log degraded details occasionally
    if ($i -in 0,5,10,20,40) {
      Log ("Health status: {0} (retriever_ready={1}, generator_ready={2}{3})" -f `
        $status, $health.retriever_ready, $health.generator_ready, `
        ($(if ($startupErr) { "; startup_error=$startupErr" } else { "" })))
    }
  } catch {
    # ignore until healthy
  }
  Start-Sleep -Seconds $HealthSleepSec
}
if (-not $healthy) {
  Log "Service did not become healthy in time."
  Log "Saving compose logs to $ComposeLogsPath"
  try { Compose @('logs') | Tee-Object -FilePath $ComposeLogsPath | Out-Null } catch {}
  throw "Health check timeout. Last startup_error: $startupErr"
}
Log "Service is healthy."

# -------- Load queries --------
if (-not (Test-Path $InputCsv)) {
  Log "Input CSV not found: $InputCsv"
  throw "Missing input CSV."
}
$rows = Import-Csv $InputCsv
$results = New-Object System.Collections.Generic.List[Object]

# -------- Run queries --------
$idx = 0; $total = $rows.Count
foreach ($row in $rows) {
  $idx++

  # detect column
  $cols = ($row | Get-Member -MemberType NoteProperty).Name
  $queryCol = ($cols | Where-Object { $_ -match '^(query|question|text)$' } | Select-Object -First 1)
  if (-not $queryCol) { $queryCol = $cols[0] }

  $q = [string]$row.$queryCol
  if ([string]::IsNullOrWhiteSpace($q)) { continue }

  $preview = $q; if ($preview.Length -gt 120) { $preview = $preview.Substring(0,120) + '...' }
  Log ("[{0}/{1}] Query: {2}" -f $idx, $total, $preview)

  $payload = @{
    query       = $q
    top_k       = 5
    temperature = 0.0
  } | ConvertTo-Json -Compress

  $resp = $null; $err = $null
  try {
    $resp = Invoke-RestMethod -Method Post -Uri $ApiUrl -ContentType "application/json" -Body $payload -TimeoutSec 180
  } catch {
    $err = $_.Exception.Message
  }

  if ($resp) {
    # sources compact/json
    $srcCompact = ""; $srcJson = ""
    if ($resp.PSObject.Properties.Name -contains 'sources' -and $resp.sources) {
      $top = @()
      foreach ($s in ($resp.sources | Select-Object -First 5)) {
        $t = ""; if ($s.text) { $t = ($s.text -replace '\s+', ' ') }
        if ($t.Length -gt 120) { $t = $t.Substring(0,120) + "..." }
        $score = ""
        if ($s.PSObject.Properties.Name -contains 'score' -and $s.score -ne $null) {
          try { $score = ("[{0:N3}] " -f [double]$s.score) } catch { $score = "" }
        }
        $src = ""; if ($s.PSObject.Properties.Name -contains 'source' -and $s.source) { $src = " (src: $($s.source))" }
        $top += ($score + $t + $src)
      }
      $srcCompact = ($top -join " | ")
      try { $srcJson = ($resp.sources | ConvertTo-Json -Compress) } catch { $srcJson = "" }
    }

    $ansPreview = ""; if ($resp.PSObject.Properties.Name -contains 'answer' -and $resp.answer) {
      $ansPreview = ($resp.answer -replace '\s+', ' ')
      if ($ansPreview.Length -gt 140) { $ansPreview = $ansPreview.Substring(0,140) + "..." }
    }
    Log ("Answer preview: {0}" -f $ansPreview)

    $ret_ms = ""; $gen_ms = ""; $tot_ms = ""
    if ($resp.PSObject.Properties.Name -contains 'timings_ms' -and $resp.timings_ms) {
      $ret_ms = $resp.timings_ms.retrieval
      $gen_ms = $resp.timings_ms.generation
      $tot_ms = $resp.timings_ms.total
    }

    $results.Add([PSCustomObject]@{
      Query               = $q
      Retrieval_Time_MS   = $ret_ms
      Generation_Time_MS  = $gen_ms
      Total_Time_MS       = $tot_ms
      Answer              = ($resp.answer | Out-String).Trim()
      Reference           = $srcCompact
      Sources_Compact     = $srcCompact
      Sources_JSON        = $srcJson
    })
  }
  else {
    Log ("ERROR: {0}" -f $err)
    $results.Add([PSCustomObject]@{
      Query               = $q
      Retrieval_Time_MS   = ""
      Generation_Time_MS  = ""
      Total_Time_MS       = ""
      Answer              = "ERROR: $err"
      Reference           = ""
      Sources_Compact     = ""
      Sources_JSON        = ""
    })
  }
}

# -------- Save & down --------
$results | Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8
Log "Saved results to $OutCsv"

Log "Stopping docker compose..."
Compose @('down') | Out-Null
$global:__stackUp = $false
Log "Test run complete."
