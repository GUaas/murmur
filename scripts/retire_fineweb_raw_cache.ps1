param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot
)

$ErrorActionPreference = "Stop"

$root = [System.IO.Path]::GetFullPath($TargetRoot)
$raw = [System.IO.Path]::GetFullPath((Join-Path $root "raw"))
$cache = [System.IO.Path]::GetFullPath((Join-Path $raw ".cache"))
$rootPrefix = $root.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$rawPrefix = $raw.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar

if (-not $raw.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Raw path is outside the target root: $raw"
}
if (-not $cache.StartsWith($rawPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Cache path is outside the raw directory: $cache"
}

$active = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and (
                $_.CommandLine -match "fineweb_edu_chinese_v21" -or
                $_.CommandLine -match "clean_fineweb"
            )
        }
)
if ($active.Count -ne 0) {
    throw "Dataset process is still active: $($active.ProcessId -join ',')"
}

$cacheFiles = @()
if (Test-Path -LiteralPath $cache -PathType Container) {
    $cacheFiles = @(
        Get-ChildItem -LiteralPath $cache -Recurse -Force -File
    )
}
$cacheBytes = ($cacheFiles | Measure-Object Length -Sum).Sum

if (Test-Path -LiteralPath $cache -PathType Container) {
    Remove-Item -LiteralPath $cache -Recurse -Force
}

$parquetDirectory = Join-Path $raw "4_5"
if (Test-Path -LiteralPath $parquetDirectory -PathType Container) {
    $parquetDirectoryItems = @(
        Get-ChildItem -LiteralPath $parquetDirectory -Force
    )
    if ($parquetDirectoryItems.Count -eq 0) {
        Remove-Item -LiteralPath $parquetDirectory -Force
    }
}

if (Test-Path -LiteralPath $raw -PathType Container) {
    $rawItems = @(Get-ChildItem -LiteralPath $raw -Force)
    if ($rawItems.Count -eq 0) {
        Remove-Item -LiteralPath $raw -Force
    }
}

[pscustomobject]@{
    deleted_cache_files = $cacheFiles.Count
    deleted_cache_bytes = [int64]$cacheBytes
    cache_exists = Test-Path -LiteralPath $cache
    raw_exists = Test-Path -LiteralPath $raw
}
