#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(git branch --show-current)" != ablation ]]; then
  echo "Refusing finalization outside ablation branch" >&2
  exit 1
fi
output="${1:-results/residual_ablation}"
output_abs="$(realpath "$output")"
repo_abs="$(pwd -P)"
[[ "$output_abs" == "$repo_abs"/results/* ]] || {
  echo "Results must be inside this repository's results directory" >&2; exit 1;
}
relative="${output_abs#"$repo_abs"/}"
[[ -s "$output_abs/summary.csv" && -s "$output_abs/REPORT.md" ]] || {
  echo "Results are incomplete; instance will stay running" >&2; exit 1;
}
while IFS= read -r changed; do
  case "$changed" in "$relative"/*) ;; *)
    echo "Uncommitted non-result change: $changed" >&2
    exit 1 ;;
  esac
done < <(git diff --name-only; git diff --cached --name-only;
         git ls-files --others --exclude-standard)

# GitHub's regular Git limit excludes giant .pt checkpoints.  Keep them on
# instance storage or publish them separately through an artifact store.
files=()
while IFS= read -r -d '' file; do
  case "$file" in *.csv|*.json|*.md|*.txt)
    if (( $(stat -c %s "$file") < 100000000 )); then files+=("$file"); fi ;;
  esac
done < <(find "$output_abs" -type f -print0)
if (( ${#files[@]} == 0 )); then
  echo "No pushable result files found" >&2; exit 1
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Dry run: ${#files[@]} result files would be committed; .pt checkpoints remain local."
  exit 0
fi
git add -f -- "${files[@]}"
if ! git diff --cached --quiet; then
  author="$(git log -1 --format=%an)"
  email="$(git log -1 --format=%ae)"
  git -c user.name="$author" -c user.email="$email" commit -m "Add completed residual A-E ablation results"
fi
git push origin ablation
local_head="$(git rev-parse HEAD)"
remote_head="$(git ls-remote origin refs/heads/ablation | cut -f1)"
[[ "$local_head" == "$remote_head" ]] || {
  echo "Remote HEAD verification failed; instance will stay running" >&2; exit 1;
}

# SSH non-login commands on Vast do not necessarily import /etc/environment.
if [[ -f /opt/supervisor-scripts/utils/environment.sh ]]; then
  # shellcheck disable=SC1091
  set +u
  . /opt/supervisor-scripts/utils/environment.sh
  set -u
fi
[[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]] || {
  echo "Push succeeded, but Vast instance credentials are unavailable; instance remains running" >&2
  exit 1
}
vastai_cli="$(command -v vastai || true)"
[[ -n "$vastai_cli" ]] || vastai_cli=/opt/instance-tools/bin/vastai
[[ -x "$vastai_cli" ]] || {
  echo "Push succeeded, but vastai CLI is missing; instance remains running" >&2
  exit 1
}
echo "Results pushed to ablation at $local_head. Stopping GPU instance $CONTAINER_ID."
"$vastai_cli" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
