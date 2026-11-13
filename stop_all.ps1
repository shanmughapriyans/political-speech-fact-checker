# =============================================================
#  stop_all.ps1  —  Stops all pipeline services
# =============================================================

function Info ($msg) { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Ok   ($msg) { Write-Host "[OK]    $msg" -ForegroundColor Green }

Info "Stopping all pipeline services..."

# Kill Python processes running our modules
Get-Process -Name "python" -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmdline -match "module[1-8]|pipeline\.py|server\.py") {
        Stop-Process -Id $_.Id -Force
        Ok "Killed: $($_.Id) — $cmdline"
    }
}

# Kill Node (dashboard)
Get-Process -Name "node" -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.Id -Force
    Ok "Killed Node process: $($_.Id)"
}

# Free ports 8765-8770 and 3000
8765,8766,8767,8768,8769,8770,3000 | ForEach-Object {
    $pids = netstat -ano 2>$null | Select-String ":$_\s" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Select-Object -Unique
    foreach ($p in $pids) {
        if ($p -match '^\d+$' -and $p -ne '0') {
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            Ok "Freed port $_ (pid $p)"
        }
    }
}

Ok "All services stopped."
