# Keep a training run alive on Windows: restart it with --resume whenever it
# exits with an error (a crash, an out-of-memory kill). A clean exit ends the
# loop. Ctrl-C checkpoints and ends it too.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_forever.ps1 runs\arena --set simulator=arena --set budget_hours=130
param([Parameter(Mandatory = $true)][string]$Run, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
python -m cfbot.train --run $Run @Rest
$status = $LASTEXITCODE
while ($status -ne 0) {
    Write-Host "training exited with status $status; resuming in 30 s"
    Start-Sleep -Seconds 30
    python -m cfbot.train --run $Run --resume
    $status = $LASTEXITCODE
}
