# tests\test_medical_rag.ps1
# Build & run with docker compose, wait for health, run queries from examples\queries_100.csv,
# log to results\log.txt, save to results\perf_results.csv (with sources), then compose down.
# PowerShell 5.1 compatible (no emojis, no ?? operator).

$apiUrl    = "http://127.0.0.1:8000/query"
$healthUrl = "http://127.0.0.1:8000/health"
$inputCsv  = "examples\queries_100.csv"
$outputDir = "results"
$outputCsv = Join-Path $outputDir "perf_results.csv"
$logPath   = Join-Path $outputDir "log.txt"

# Ensure output dir
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
"" | Out-File -FilePath $logPath -Encoding UTF8  # reset log

function Log($msg) {
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$stamp] $msg"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

Log "Starting docker compose (detached, build if needed)..."
docker compose up -d --build | Out-Null

# Wait for health
Log "Waiting for $healthUrl ..."
$maxRetries = 40
$healthy = $false
for ($i=0; $i -lt $maxRetries; $i++) {
    try {
        $health = Invoke-RestMethod $healthUrl -TimeoutSec 3
        if ($health.status -eq "ok") {
            $healthy = $true
            break
        }
    } catch {
        # ignore until healthy
    }
    Start-Sleep -Seconds 3
}
if (-not $healthy) {
    Log "Service did not become healthy in time. Saving logs and exiting."
    docker compose logs | Tee-Object -FilePath (Join-Path $outputDir "compose_logs.txt")
    docker compose down | Out-Null
    exit 1
}
Log "Service is healthy."

# Load queries
if (-not (Test-Path $inputCsv)) {
    Log "Input CSV not found: $inputCsv"
    docker compose down | Out-Null
    exit 1
}
$rows = Import-Csv $inputCsv

# Prepare output
$results = New-Object System.Collections.Generic.List[Object]
$idx = 0
$total = $rows.Count

foreach ($row in $rows) {
    $idx++

    # Pick query column (query/question/text or first col)
    $cols = ($row | Get-Member -MemberType NoteProperty).Name
    $queryCol = ($cols | Where-Object { $_ -match '^(query|question|text)$' } | Select-Object -First 1)
    if (-not $queryCol) { $queryCol = $cols[0] }

    $q = [string]$row.$queryCol
    if ([string]::IsNullOrWhiteSpace($q)) { continue }

    $preview = $q
    if ($preview.Length -gt 120) { $preview = $preview.Substring(0,120) + "..." }
    Log ("[{0}/{1}] Query: {2}" -f $idx, $total, $preview)

    $payload = @{
        query = $q
        k = 5
        temperature = 0.0
    } | ConvertTo-Json -Compress

    $resp = $null
    $err  = $null
    try {
        $resp = Invoke-RestMethod -Method Post -Uri $apiUrl -ContentType "application/json" -Body $payload -TimeoutSec 120
    } catch {
        $err = $_.Exception.Message
    }

    if ($resp) {
        # Build compact sources and raw JSON
        $srcCompact = ""
        $srcJson = ""
        if ($resp.sources) {
            $top = @()
            foreach ($s in ($resp.sources | Select-Object -First 5)) {
                $t = ""
                if ($s.text) { $t = ($s.text -replace '\s+', ' ') }
                if ($t.Length -gt 120) { $t = $t.Substring(0,120) + "..." }
                $score = ""
                if ($s.score -ne $null) {
                    try { $score = ("[{0:N3}] " -f [double]$s.score) } catch { $score = "" }
                }
                $src = ""
                if ($s.source) { $src = " (src: $($s.source))" }
                $top += ($score + $t + $src)
            }
            $srcCompact = ($top -join " | ")
            try { $srcJson = ($resp.sources | ConvertTo-Json -Compress) } catch { $srcJson = "" }
        }

        # Answer preview to log
        $ansPreview = $resp.answer
        if (-not $ansPreview) { $ansPreview = "" }
        $ansPreview = ($ansPreview -replace '\s+', ' ')
        if ($ansPreview.Length -gt 140) { $ansPreview = $ansPreview.Substring(0,140) + "..." }
        Log ("Answer preview: {0}" -f $ansPreview)

        # Timings (safe access)
        $ret_ms = ""
        $gen_ms = ""
        $tot_ms = ""
        if ($resp.PSObject.Properties.Name -contains "timings_ms" -and $resp.timings_ms) {
            $ret_ms = $resp.timings_ms.retrieval
            $gen_ms = $resp.timings_ms.generation
            $tot_ms = $resp.timings_ms.total
        }

        $results.Add([PSCustomObject]@{
            Query               = $q
            Retrieval_Time_MS   = $ret_ms
            Generation_Time_MS  = $gen_ms
            Total_Time_MS       = $tot_ms
            Answer              = $resp.answer
            Reference           = $srcCompact   # <-- use retrieved sources as reference
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
            Reference           = ""           # Keep column consistent for errors
            Sources_Compact     = ""
            Sources_JSON        = ""
        })
    }
}

# Save CSV
$results | Export-Csv -Path $outputCsv -NoTypeInformation -Encoding UTF8
Log "Saved results to $outputCsv"

# Bring the stack down
Log "Stopping docker compose..."
docker compose down | Out-Null
Log "Test run complete."
