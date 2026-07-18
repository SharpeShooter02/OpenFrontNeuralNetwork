#!/usr/bin/env bash
# OpenFrontNN Phase 0 setup (macOS/Linux)
# Usage:  bash scripts/setup.sh
set -euo pipefail

ENGINE_COMMIT="774d98ddad3123a7d12709ae029c853aea1b0039"

[ -d .git ] || git init

if [ ! -d vendor/OpenFrontIO/.git ]; then
  # --depth 1 = shallow clone: skips the heavy binary-asset history
  git submodule add --depth 1 https://github.com/openfrontio/OpenFrontIO.git vendor/OpenFrontIO
fi
git -C vendor/OpenFrontIO fetch --depth 1 origin "$ENGINE_COMMIT"
git -C vendor/OpenFrontIO checkout "$ENGINE_COMMIT"

# engine deps (skip husky prepare hook, which fails outside its own repo)
( cd vendor/OpenFrontIO && npm install --ignore-scripts )

# our dev deps (tsx)
npm install

echo ""
echo "Setup complete. Run:  npm run harness"
