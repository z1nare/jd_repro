#!/usr/bin/env bash
# Run the full gate suite. This is the preflight: if it does not pass, nothing
# downstream is trustworthy.
#
# PLACEHOLDER -- extend as gates land. Right now only the legacy CIFAR gate is
# implemented; the 5x gates skip themselves.
set -euo pipefail

python -m pytest gates/ -v
