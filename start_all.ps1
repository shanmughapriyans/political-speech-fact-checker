# =============================================================
#  start_all.ps1  —  One-click launcher for the full pipeline
# =============================================================
#  Usage:
#    .\start_all.ps1                      # microphone (live feed)
#    .\start_all.ps1 -File path\to\audio.wav  # audio file
#    .\start_all.ps1 -Model tiny          # fastest (least accurate)
#    .\start_all.ps1 -Model base          # fast + accurate (DEFAULT)
#    .\start_all.ps1 -Model small         # more accurate (needs GPU)
# =============================================================

param(
    [string]$File  = "",          # leave empty for microphone input
    [string]$Model = "base"       # whisper model: tiny | base | small | medium
)

$Root    = $PSScriptRoot
$Venv1   = "$Root\module1_speech_asr\venv\Scripts\python.exe"
$VenvM   = "$Root\module2_claim_detection\venv\Scripts\python.exe"

# ---- Colour helpers ----
function Info  ($msg) { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Ok    ($msg) { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Warn  ($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Banner($msg) { Write-Host "`n========================================" -ForegroundColor Magenta
                        Write-Host "  $msg" -ForegroundColor Magenta
                        Write-Host "========================================`n" -ForegroundColor Magenta }

Banner "Fact Verification System — Starting All Modules"

# ---- Kill any old instances on our ports ----
Info "Clearing any stale processes on ports 8765-8770..."
8765,8766,8767,8768,8769,8770 | ForEach-Object {
    $pids = netstat -ano 2>$null | Select-String ":$_\s" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Select-Object -Unique
    foreach ($p in $pids) {
        if ($p -match '^\d+$' -and $p -ne '0') {
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        }
    }
}
Start-Sleep -Seconds 1

# ---- Module 2: Claim Detection (ws://8766) ----
Info "Starting Module 2 — Claim Detection (ws://8766)..."
Start-Process -FilePath $VenvM `
    -ArgumentList "$Root\module2_claim_detection\pipeline.py" `
    -WorkingDirectory "$Root\module2_claim_detection" `
    -WindowStyle Minimized
Start-Sleep -Seconds 2
Ok "Module 2 started."

# ---- Module 3: Query Formulation (ws://8767) ----
Info "Starting Module 3 — Query Formulation (ws://8767)..."
Start-Process -FilePath $VenvM `
    -ArgumentList "$Root\module3_query_formulation\server.py" `
    -WorkingDirectory "$Root\module3_query_formulation" `
    -WindowStyle Minimized
Start-Sleep -Seconds 2
Ok "Module 3 started."

# ---- Module 4: Evidence Retrieval (ws://8768) ----
Info "Starting Module 4 — Evidence Retrieval (ws://8768)..."
Start-Process -FilePath $VenvM `
    -ArgumentList "$Root\module4_evidence_retrieval\server.py" `
    -WorkingDirectory "$Root\module4_evidence_retrieval" `
    -WindowStyle Minimized
Start-Sleep -Seconds 3
Ok "Module 4 started (BM25 index loading...)."

# ---- Module 5: RAG Verification (ws://8769) ----
Info "Starting Module 5 — RAG Verification (ws://8769)..."
Start-Process -FilePath $VenvM `
    -ArgumentList "$Root\module5_rag_verification\server.py" `
    -WorkingDirectory "$Root\module5_rag_verification" `
    -WindowStyle Minimized
Start-Sleep -Seconds 3
Ok "Module 5 started (DeBERTa loading...)."

# ---- Module 6: Verdict Resolution (ws://8770) ----
Info "Starting Module 6 — Verdict Resolution (ws://8770)..."
Start-Process -FilePath $VenvM `
    -ArgumentList "$Root\module6_verdict_resolution\server.py" `
    -WorkingDirectory "$Root\module6_verdict_resolution" `
    -WindowStyle Minimized
Start-Sleep -Seconds 2
Ok "Module 6 started."

# ---- Module 7: Next.js Dashboard (http://localhost:3000) ----
Info "Starting Module 7 — Dashboard (http://localhost:3000)..."
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c npm run dev" `
    -WorkingDirectory "$Root\module7_dashboard" `
    -WindowStyle Minimized
Start-Sleep -Seconds 5
Ok "Module 7 started. Open http://localhost:3000 in your browser."

# ---- Wait for all services to be ready ----
Info "Waiting for all WebSocket services to be ready..."
Start-Sleep -Seconds 5

# ---- Module 1: ASR — LAST (triggers the whole pipeline) ----
Banner "Starting Module 1 — ASR (this triggers the pipeline!)"

if ($File -ne "") {
    # --- FILE MODE ---
    if (-not (Test-Path $File)) {
        Warn "File not found: $File"
        Warn "Falling back to microphone input."
        $File = ""
    }
}

if ($File -ne "") {
    Info "Input: FILE  → $File"
    Info "Model: $Model  (beam_size=1 for speed)"
    Info "Transcription will begin immediately..."
    Write-Host ""
    & $Venv1 "$Root\module1_speech_asr\server.py" --input file --file $File --model $Model
} else {
    Info "Input: MICROPHONE (live feed)"
    Info "Model: $Model  (beam_size=1 for speed)"
    Info "Speak into your microphone — transcription starts immediately."
    Info "Press Ctrl+C to stop."
    Write-Host ""
    & $Venv1 "$Root\module1_speech_asr\server.py" --input mic --model $Model
}

Banner "Pipeline stopped. All background modules still running."
Info "To stop everything, close the minimized windows or restart PowerShell."
