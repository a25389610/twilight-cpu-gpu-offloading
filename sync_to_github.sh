#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

commit_message="${*:-Update reports scripts and source}"
remote_name="origin"
branch="$(git branch --show-current)"

if [[ "$branch" != "main" ]]; then
  echo "Error: expected branch main, found '$branch'." >&2
  exit 1
fi

if ! git remote get-url "$remote_name" >/dev/null 2>&1; then
  echo "Error: remote '$remote_name' is not configured." >&2
  exit 1
fi

echo "Repository: $repo_dir"
echo "Remote:    $(git remote get-url "$remote_name")"
echo "Branch:    $branch"
echo

echo "Changes that will be included:"
git status --short -- reports scripts source CURRENT_IMPLEMENTATION.md sync_to_github.sh

if git diff --quiet -- reports scripts source CURRENT_IMPLEMENTATION.md sync_to_github.sh \
  && git diff --cached --quiet -- reports scripts source CURRENT_IMPLEMENTATION.md sync_to_github.sh; then
  echo
  echo "No changes found in reports/, scripts/, source/, or CURRENT_IMPLEMENTATION.md."
  exit 0
fi

git add -- reports scripts source CURRENT_IMPLEMENTATION.md sync_to_github.sh

echo
echo "Staged changes:"
git diff --cached --stat
echo
read -r -p "Commit and push these changes to $remote_name/$branch? [y/N] " answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  git reset >/dev/null
  echo "Cancelled. No commit or push was performed."
  exit 0
fi

git commit -m "$commit_message"
git push "$remote_name" "$branch"

local_commit="$(git rev-parse HEAD)"
remote_commit="$(git ls-remote "$remote_name" "refs/heads/$branch" | awk '{print $1}')"

echo
if [[ "$local_commit" == "$remote_commit" ]]; then
  echo "Verified: GitHub is synchronized at $local_commit."
else
  echo "Warning: local commit ($local_commit) differs from remote ($remote_commit)." >&2
  exit 1
fi

git status --short
