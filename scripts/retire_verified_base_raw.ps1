param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot,

    [string]$BaseManifestName = "download_combined_manifest.json",
    [string]$RemainderManifestName = "download_remainder_manifest.json",
    [string]$CleaningReportName = "cleaning_report.json",
    [string]$CleanedDirectoryName = "cleaned",
    [string]$RetirementReportName = "base_raw_retirement_report.json",
    [int]$ExpectedSelectionFiles = 4900,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

function Read-JsonObject {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description does not exist: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Resolve-ChildFilePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Candidate,

        [Parameter(Mandatory = $true)]
        [string]$AllowedRoot
    )

    $resolvedCandidate = [System.IO.Path]::GetFullPath($Candidate)
    $resolvedRoot = [System.IO.Path]::GetFullPath($AllowedRoot)
    $rootPrefix = $resolvedRoot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedCandidate.StartsWith(
        $rootPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing path outside the allowed raw directory: $resolvedCandidate"
    }
    return $resolvedCandidate
}

function Write-JsonAtomically {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [object]$Payload
    )

    $temporaryPath = "$Path.tmp-$PID"
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $temporaryPath,
        ($Payload | ConvertTo-Json -Depth 8),
        $encoding
    )
    Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
}

$root = [System.IO.Path]::GetFullPath($TargetRoot)
$rawRoot = Join-Path $root "raw"
$cleanedRoot = Join-Path $root $CleanedDirectoryName
$baseManifestPath = Join-Path $root $BaseManifestName
$remainderManifestPath = Join-Path $root $RemainderManifestName
$cleaningReportPath = Join-Path $root $CleaningReportName
$retirementReportPath = Join-Path $root $RetirementReportName

$baseManifest = Read-JsonObject `
    -Path $baseManifestPath `
    -Description "Base download manifest"
$remainderManifest = Read-JsonObject `
    -Path $remainderManifestPath `
    -Description "Remainder download manifest"
$cleaningReport = Read-JsonObject `
    -Path $cleaningReportPath `
    -Description "Base cleaning report"

if ($baseManifest.status -ne "completed") {
    throw "Base download manifest is not completed: $($baseManifest.status)"
}
if ([int]$baseManifest.completed_count -ne $ExpectedSelectionFiles) {
    throw (
        "Expected {0} completed selected files, got {1}" -f `
            $ExpectedSelectionFiles,
            $baseManifest.completed_count
    )
}
if (@($baseManifest.failures).Count -ne 0) {
    throw "Base download manifest contains failures"
}
if ($cleaningReport.status -ne "completed") {
    throw "Base cleaning report is not completed: $($cleaningReport.status)"
}

$baseRepoPaths = @($baseManifest.selected_repo_paths | ForEach-Object { [string]$_ })
$remainderRepoPaths = @(
    $remainderManifest.selected_repo_paths | ForEach-Object { [string]$_ }
)
$basePathSet = New-Object "System.Collections.Generic.HashSet[string]"
foreach ($repoPath in $baseRepoPaths) {
    if (-not $basePathSet.Add($repoPath)) {
        throw "Base manifest contains a duplicate repository path: $repoPath"
    }
}
$overlap = @(
    $remainderRepoPaths |
        Where-Object { $basePathSet.Contains($_) }
)
if ($overlap.Count -ne 0) {
    throw "Base and remainder selections overlap on $($overlap.Count) files"
}

$expectedShards = @($cleaningReport.output_shards)
if ($expectedShards.Count -eq 0) {
    throw "Cleaning report does not list output shards"
}

$verifiedCleanedBytes = [int64]0
$hashStarted = Get-Date
for ($index = 0; $index -lt $expectedShards.Count; $index++) {
    $shard = $expectedShards[$index]
    $shardName = [string]$shard.file
    if ([System.IO.Path]::GetFileName($shardName) -ne $shardName) {
        throw "Invalid cleaned shard file name: $shardName"
    }
    $shardPath = Join-Path $cleanedRoot $shardName
    if (-not (Test-Path -LiteralPath $shardPath -PathType Leaf)) {
        throw "Cleaned shard is missing: $shardPath"
    }
    $item = Get-Item -LiteralPath $shardPath
    if ([int64]$item.Length -ne [int64]$shard.size_bytes) {
        throw "Cleaned shard size mismatch: $shardPath"
    }
    $actualHash = (Get-FileHash -LiteralPath $shardPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedHash = ([string]$shard.sha256).ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Cleaned shard SHA-256 mismatch: $shardPath"
    }
    $verifiedCleanedBytes += [int64]$item.Length
    if (($index + 1) % 8 -eq 0 -or ($index + 1) -eq $expectedShards.Count) {
        Write-Output (
            "verified_cleaned_shards={0}/{1} verified_GiB={2:N3}" -f `
                ($index + 1),
                $expectedShards.Count,
                ($verifiedCleanedBytes / 1GB)
        )
    }
}
if ($verifiedCleanedBytes -ne [int64]$cleaningReport.output_bytes) {
    throw (
        "Verified cleaned bytes do not match cleaning report: actual={0}, expected={1}" -f `
            $verifiedCleanedBytes,
            [int64]$cleaningReport.output_bytes
    )
}

$baseRecordsByRepoPath = @{}
foreach ($record in @($baseManifest.completed_files)) {
    $baseRecordsByRepoPath[[string]$record.repo_path] = $record
}
if ($baseRecordsByRepoPath.Count -ne $ExpectedSelectionFiles) {
    throw (
        "Expected {0} unique completed selected records, got {1}" -f `
            $ExpectedSelectionFiles,
            $baseRecordsByRepoPath.Count
    )
}

$resolvedBaseFiles = New-Object System.Collections.Generic.List[object]
$preexistingMissing = 0
$reclaimableBytes = [int64]0
foreach ($repoPath in $baseRepoPaths) {
    if (-not $baseRecordsByRepoPath.ContainsKey($repoPath)) {
        throw "Base manifest lacks a completed record for: $repoPath"
    }
    $record = $baseRecordsByRepoPath[$repoPath]
    $localPath = Resolve-ChildFilePath `
        -Candidate ([string]$record.local_path) `
        -AllowedRoot $rawRoot
    $expectedBytes = [int64]$record.size_bytes
    if (Test-Path -LiteralPath $localPath -PathType Leaf) {
        $actualBytes = [int64](Get-Item -LiteralPath $localPath).Length
        if ($actualBytes -ne $expectedBytes) {
            throw "Base raw file size mismatch: $localPath"
        }
        $reclaimableBytes += $actualBytes
    }
    else {
        $preexistingMissing += 1
    }
    $resolvedBaseFiles.Add(
        [pscustomobject]@{
            repo_path = $repoPath
            local_path = $localPath
            size_bytes = $expectedBytes
        }
    )
}

$plan = [ordered]@{
    schema_version = 1
    status = if ($Execute) { "verified_ready_to_delete" } else { "verified_dry_run" }
    created_at = (Get-Date).ToString("o")
    target_root = $root
    base_manifest = $BaseManifestName
    remainder_manifest = $RemainderManifestName
    cleaning_report = $CleaningReportName
    cleaned_shards_verified = $expectedShards.Count
    cleaned_bytes_verified = $verifiedCleanedBytes
    cleaned_hash_elapsed_seconds = [math]::Round(
        ((Get-Date) - $hashStarted).TotalSeconds,
        3
    )
    base_files_selected = $baseRepoPaths.Count
    base_files_preexisting_missing = $preexistingMissing
    base_files_to_delete = $resolvedBaseFiles.Count - $preexistingMissing
    reclaimable_bytes = $reclaimableBytes
    remainder_files_selected = $remainderRepoPaths.Count
    selection_overlap = 0
}

if (-not $Execute) {
    Write-Output ($plan | ConvertTo-Json -Depth 4)
    exit 0
}

$deletedFiles = 0
$deletedBytes = [int64]0
foreach ($record in $resolvedBaseFiles) {
    if (-not (Test-Path -LiteralPath $record.local_path -PathType Leaf)) {
        continue
    }
    Remove-Item -LiteralPath $record.local_path -Force
    if (Test-Path -LiteralPath $record.local_path) {
        throw "Base raw file still exists after deletion: $($record.local_path)"
    }
    $deletedFiles += 1
    $deletedBytes += [int64]$record.size_bytes
    if ($deletedFiles % 250 -eq 0) {
        Write-Output (
            "deleted_base_files={0}/{1} released_GiB={2:N3}" -f `
                $deletedFiles,
                $plan.base_files_to_delete,
                ($deletedBytes / 1GB)
        )
    }
}

$remainingBaseFiles = @(
    $resolvedBaseFiles |
        Where-Object { Test-Path -LiteralPath $_.local_path -PathType Leaf }
)
if ($remainingBaseFiles.Count -ne 0) {
    throw "Failed to retire $($remainingBaseFiles.Count) base raw files"
}

$plan.status = "completed"
$plan.completed_at = (Get-Date).ToString("o")
$plan.deleted_files = $deletedFiles
$plan.deleted_bytes = $deletedBytes
$plan.remaining_base_files = 0
Write-JsonAtomically -Path $retirementReportPath -Payload $plan
Write-Output ($plan | ConvertTo-Json -Depth 4)
