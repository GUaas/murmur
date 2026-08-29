param(
    [Parameter(Mandatory = $true, ParameterSetName = "Text")]
    [string]$Text,

    [Parameter(Mandatory = $true, ParameterSetName = "File")]
    [string]$InputFile,

    [string]$OutputFile,

    [ValidateSet("auto", "always", "never")]
    [string]$LongTextMode = "auto",

    [int]$ChunkTokens = 160,

    [switch]$ShowStats
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $ProjectRoot
$Arguments = @(
    (Join-Path $ProjectRoot "scripts\simplify_text.py"),
    "--config", (Join-Path $ProjectRoot "configs\inference_text_simplification_203m.yaml"),
    "--long-text-mode", $LongTextMode,
    "--chunk-tokens", $ChunkTokens
)
if ($PSCmdlet.ParameterSetName -eq "File") {
    $Arguments += @("--text-file", $InputFile)
} else {
    $Arguments += @("--text", $Text)
}
if ($OutputFile) {
    $Arguments += @("--output-file", $OutputFile)
}
if ($ShowStats) {
    $Arguments += "--show-stats"
}
python @Arguments
