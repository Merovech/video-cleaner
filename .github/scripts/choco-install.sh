#!/usr/bin/env bash
# Install Chocolatey packages, retrying while the community feed is down.
#
# Usage: choco-install.sh PACKAGE...
#
# During a feed outage (HTTP 503) `choco install` can report "installed 0/0 packages" and still
# exit 0, so success is judged by the list of installed packages, not by the exit code.
# CHOCO_ATTEMPTS (default 3) and CHOCO_RETRY_DELAY_S (default 20) tune the retries.
set -u

attempts="${CHOCO_ATTEMPTS:-3}"
delay="${CHOCO_RETRY_DELAY_S:-20}"

# Print the requested packages that choco does not list as installed.
missing_packages() {
  local installed pkg
  installed="$(choco list --limit-output)" || installed=""
  for pkg in "$@"; do
    printf '%s\n' "$installed" | grep -qi "^${pkg}|" || echo "$pkg"
  done
}

missing=""
for attempt in $(seq 1 "$attempts"); do
  choco install "$@" -y --no-progress
  missing="$(missing_packages "$@")"
  if [ -z "$missing" ]; then
    exit 0
  fi
  echo "attempt ${attempt}/${attempts}: not installed: ${missing//$'\n'/ }" >&2
  if [ "$attempt" -lt "$attempts" ]; then
    sleep "$delay"
  fi
done
echo "::error::could not install: ${missing//$'\n'/ }" >&2
exit 1
