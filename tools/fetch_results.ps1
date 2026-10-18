<#
.SYNOPSIS
  Downloads the results streamed by the Simulate workflow to the release sim-<Tag> and unpacks them.

.DESCRIPTION
  Polls the release every -IntervalMinutes. New archives (and latest_segNN.tar.gz whenever it is
  re-uploaded) are downloaded with gh and unpacked into -Dest, so the folder has the usual layout:
  res/{IS,Fl,Or,Ne,IO}/Z*.dat, res/Imax.dat, ..., SpSIC.dat, Incident.dat, log_segNN.txt.
  What was already unpacked is remembered in <Dest>/.downloaded.json, so the script can be stopped
  and started again at any time. It exits once the release contains DONE.txt.
  kw_THz is not in the release: it is in the artifacts kw-<Tag>-segNN of every segment run.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\fetch_results.ps1 -Tag z30-old
#>
param(
  [Parameter(Mandatory = $true)] [string]$Tag,
  # Default: UPPE_<Tag> next to the repository folder.
  [string]$Dest,
  [double]$IntervalMinutes = 5,
  [string]$Repo = 'NailKhas/UPPE_THZ_DC_cpp',
  [double]$MinFreeGB = 0.5
)

# Not 'Stop': Windows PowerShell 5.1 turns any stderr output of gh or tar into a terminating error,
# even with 2>$null. Every native call is checked through $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
$release = "sim-$Tag"
if (-not $Dest) {
  $Dest = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "UPPE_$Tag"
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$Dest = (Resolve-Path $Dest).ProviderPath
$incoming = Join-Path $Dest '.incoming'
New-Item -ItemType Directory -Force -Path $incoming | Out-Null
$stateFile = Join-Path $Dest '.downloaded.json'

$state = @{}
if (Test-Path $stateFile) {
  (Get-Content -Raw $stateFile | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $state[$_.Name] = $_.Value }
}

function Save-State {
  $state | ConvertTo-Json | Set-Content -Encoding UTF8 $stateFile
}

function Write-Log([string]$msg) {
  Write-Host ("[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg)
}

function Get-FreeGB {
  $drive = (Split-Path -Qualifier $Dest).TrimEnd(':')
  return (Get-PSDrive $drive).Free / 1GB
}

function Get-Assets {
  $json = & gh release view $release --repo $Repo --json assets 2>$null
  if ($LASTEXITCODE -ne 0) { return $null }
  # The comma keeps an empty asset list from being unrolled into $null.
  return , @(($json | Out-String | ConvertFrom-Json).assets)
}

function Show-Status {
  $imax = Join-Path $Dest 'res\Imax.dat'
  $z = '?'
  if (Test-Path $imax) {
    $last = Get-Content $imax -Tail 1
    if ($last) { $z = ($last.Trim() -split '\s+')[0] }
  }
  $run = & gh run list --repo $Repo --workflow simulate.yml -L 1 --json status,conclusion,displayTitle 2>$null
  $runText = ''
  if ($LASTEXITCODE -eq 0 -and $run) {
    $r = @($run | Out-String | ConvertFrom-Json)
    if ($r.Count -gt 0) { $runText = "; last run: $($r[0].displayTitle) [$($r[0].status) $($r[0].conclusion)]" }
  }
  Write-Log "z = $z cm$runText"
}

Write-Log "Fetching release $release of $Repo into $Dest every $IntervalMinutes min"
while ($true) {
  $assets = Get-Assets
  if ($null -eq $assets) {
    Write-Log "release $release is not available yet (or gh failed); retrying"
  } else {
    $done = $false
    foreach ($a in ($assets | Sort-Object name)) {
      if ($a.name -eq 'DONE.txt') { $done = $true; continue }
      if (-not $a.name.EndsWith('.tar.gz')) { continue }
      # PowerShell 7 turns ISO dates into DateTime, 5.1 does not; compare as strings either way.
      $stamp = "$([string]$a.updatedAt)|$($a.size)"
      if ([string]$state[$a.name] -eq $stamp) { continue }

      $needGB = 3 * $a.size / 1GB + $MinFreeGB
      $freeGB = Get-FreeGB
      if ($freeGB -lt $needGB) {
        Write-Log ("not enough disk space for {0}: {1:N2} GB free, {2:N2} GB needed; skipped for now" -f $a.name, $freeGB, $needGB)
        continue
      }

      $file = Join-Path $incoming $a.name
      & gh release download $release --repo $Repo --pattern $a.name --dir $incoming --clobber 2>&1 | Out-Null
      if ($LASTEXITCODE -ne 0 -or -not (Test-Path $file)) {
        Write-Log "download of $($a.name) failed; will retry"
        continue
      }
      & tar.exe -xzf $file -C $Dest
      if ($LASTEXITCODE -ne 0) {
        Write-Log "unpacking $($a.name) failed; will retry"
        continue
      }
      Remove-Item $file
      $state[$a.name] = $stamp
      Save-State
      Write-Log ("unpacked {0} ({1:N1} MB)" -f $a.name, ($a.size / 1MB))
    }
    Show-Status
    if ($done) {
      & gh release download $release --repo $Repo --pattern 'DONE.txt' --dir $Dest --clobber 2>&1 | Out-Null
      Write-Log "the chain is finished: $(Get-Content (Join-Path $Dest 'DONE.txt') -ErrorAction SilentlyContinue)"
      break
    }
  }
  Start-Sleep -Seconds ([int]($IntervalMinutes * 60))
}
