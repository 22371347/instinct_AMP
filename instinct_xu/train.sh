#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$package_root"
export PYTHONPATH="$package_root/source/instinctlab:$package_root/source/instinct_rl${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/instinct_rl/train.py "$@"
