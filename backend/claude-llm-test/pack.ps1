# Claude LLM 테스트 zip 패키지 생성
# 사용: backend/claude-llm-test 폴더에서  .\pack.ps1

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$backend = Split-Path $here -Parent
$outName = "claude-llm-test"
$staging = Join-Path $env:TEMP $outName
$zipPath = Join-Path $here "$outName.zip"

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

New-Item -ItemType Directory -Path $staging | Out-Null

$files = @(
    "test_claude.py",
    "test_llm_helpers.py",
    ".env.example",
    "models\__init__.py",
    "models\schemas.py",
    "pipeline\__init__.py",
    "pipeline\stage5_claude.py",
    "pipeline\stage5_common.py",
    "pipeline\pre_stage.py",
    "pipeline\stage0_extract.py",
    "pipeline\stage2_ocr.py",
    "pipeline\stage4_embedding.py",
    "pipeline\stage1_evidence.py",
    "config\__init__.py",
    "config\keywords.json",
    "config\loader.py",
    "db\__init__.py",
    "db\cache.py"
)

foreach ($rel in $files) {
    $src = Join-Path $backend $rel
    if (-not (Test-Path $src)) {
        Write-Error "Missing: $src"
    }
    $dest = Join-Path $staging $rel
    $destDir = Split-Path $dest -Parent
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    Copy-Item $src $dest
}

Copy-Item (Join-Path $here "requirements.txt") (Join-Path $staging "requirements.txt")
Copy-Item (Join-Path $here "TEST_MANUAL.md") (Join-Path $staging "TEST_MANUAL.md")

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -Force
Remove-Item $staging -Recurse -Force

Write-Host "Created: $zipPath" -ForegroundColor Green
Write-Host "Share this zip with your team (do NOT include .env or API keys)."
