param(
    [string]$RepoId = "hunshui01/murmur-fineweb-edu-chinese-v21-sp32k",
    [string]$ProjectRoot = "",
    [string]$CacheRoot = "D:\datasets\fineweb_edu_chinese_v21_score45_12b\token_cache_sp32k_2048",
    [int]$MaxWorkers = 4
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Invoke-ModelScopeUpload {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [Parameter(Mandatory = $true)][string]$CommitMessage,
        [int]$Workers = 1
    )

    if (-not (Test-Path -LiteralPath $LocalPath)) {
        throw "Local upload source does not exist: $LocalPath"
    }

    Write-Output "UPLOAD_START remote=$RemotePath local=$LocalPath"
    & python -m modelscope.cli.cli upload `
        $RepoId `
        $LocalPath `
        $RemotePath `
        --repo-type dataset `
        --revision master `
        --commit-message $CommitMessage `
        --max-workers $Workers `
        --use-cache `
        --disable-tqdm

    if ($LASTEXITCODE -ne 0) {
        throw "ModelScope upload failed for $RemotePath with exit code $LASTEXITCODE"
    }
    Write-Output "UPLOAD_COMPLETED remote=$RemotePath"
}

$cacheManifest = Join-Path $CacheRoot "manifest.json"
$validationReport = Join-Path $CacheRoot "validation_report.json"
if (-not (Test-Path -LiteralPath $cacheManifest)) {
    throw "Cache manifest is missing: $cacheManifest"
}
if (-not (Test-Path -LiteralPath $validationReport)) {
    throw "Validation report is missing: $validationReport"
}

Invoke-ModelScopeUpload `
    -LocalPath $CacheRoot `
    -RemotePath "token_cache_sp32k_2048" `
    -CommitMessage "Upload audited 17.85B-token cache" `
    -Workers $MaxWorkers

Invoke-ModelScopeUpload `
    -LocalPath (Join-Path $ProjectRoot "tokenizer\sp_unigram_32k.model") `
    -RemotePath "tokenizer/sp_unigram_32k.model" `
    -CommitMessage "Add 32K SentencePiece Unigram tokenizer"

Invoke-ModelScopeUpload `
    -LocalPath (Join-Path $ProjectRoot "configs\pretrain_fineweb_v21_17.84b_554m_deep_sp32k.yaml") `
    -RemotePath "configs/pretrain_fineweb_v21_17.84b_554m_deep_sp32k.yaml" `
    -CommitMessage "Add Murmur 554M pretraining config"

Invoke-ModelScopeUpload `
    -LocalPath (Join-Path $ProjectRoot "configs\pretrain_fineweb_v21_2.10m_554m_deep_sp32k_smoke.yaml") `
    -RemotePath "configs/pretrain_fineweb_v21_2.10m_554m_deep_sp32k_smoke.yaml" `
    -CommitMessage "Add Murmur 554M smoke-test config"

Invoke-ModelScopeUpload `
    -LocalPath (Join-Path $ProjectRoot "MODELSCOPE_DATASET_README.md") `
    -RemotePath "README.md" `
    -CommitMessage "Add dataset card and usage notes"

Write-Output "DATASET_UPLOAD_COMPLETED repo=$RepoId"
