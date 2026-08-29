param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId,

    [Parameter(Mandatory = $true)]
    [string]$TargetRoot,

    [Parameter(Mandatory = $true)]
    [string]$Revision,

    [int]$TargetTotalFiles = 4900,
    [int]$Workers = 6,
    [int]$CheckpointInterval = 10,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$extensionScript = Join-Path $PSScriptRoot "extend_fineweb_edu_chinese_v21.py"
$logPath = Join-Path $TargetRoot "download_extension.log"
$errorLogPath = Join-Path $TargetRoot "download_extension.error.log"

Wait-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue

& $PythonExecutable -u $extensionScript `
    --target-root $TargetRoot `
    --revision $Revision `
    --target-total-files $TargetTotalFiles `
    --workers $Workers `
    --checkpoint-interval $CheckpointInterval `
    1>> $logPath 2>> $errorLogPath

exit $LASTEXITCODE
