# Start the from-scratch detector ML backend. Run in its OWN terminal AFTER env.ps1.
#
#   . .\scripts\env.ps1
#   .\scripts\start_ml_backend.ps1

$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\Activate.ps1"

$port = "9090"
$lanip = $env:LAN_IP
if (-not $lanip) {
  $lanip = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.NetAdapter.Status -eq 'Up' } |
            Select-Object -First 1).IPv4Address.IPAddress
}
if (-not $lanip) { $lanip = "localhost" }

Set-Location "$root\ml_backend"
Write-Host "Starting from-scratch detector ML backend - served on the local network" -ForegroundColor Green
Write-Host "  this machine : http://localhost:$port"
Write-Host "  on your LAN  : http://${lanip}:$port" -ForegroundColor Cyan
Write-Host "  firewall (Admin) if other devices can't reach it:" -ForegroundColor DarkGray
Write-Host "    New-NetFirewallRule -DisplayName 'ML Backend $port' -Direction Inbound -LocalPort $port -Protocol TCP -Action Allow -Profile Private" -ForegroundColor DarkGray
python _wsgi.py
