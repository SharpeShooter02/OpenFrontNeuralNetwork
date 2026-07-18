# OpenFrontNN Phase 0 setup (Windows PowerShell)
# Usage:  ./scripts/setup.ps1
$ErrorActionPreference = "Stop"

$ENGINE_COMMIT = "774d98ddad3123a7d12709ae029c853aea1b0039"

if (-not (Test-Path ".git")) { git init }

if (-not (Test-Path "vendor/OpenFrontIO/.git")) {
  # --depth 1 = shallow clone: skips the heavy binary-asset history
  git submodule add --depth 1 https://github.com/openfrontio/OpenFrontIO.git vendor/OpenFrontIO
}
git -C vendor/OpenFrontIO fetch --depth 1 origin $ENGINE_COMMIT
git -C vendor/OpenFrontIO checkout $ENGINE_COMMIT

# engine deps (skip husky prepare hook, which fails outside its own repo)
Push-Location vendor/OpenFrontIO
npm install --ignore-scripts
Pop-Location

# our dev deps (tsx)
npm install

Write-Host "`nSetup complete. Run:  npm run harness" -ForegroundColor Green
