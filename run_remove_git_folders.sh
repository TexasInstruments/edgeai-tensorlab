#!/usr/bin/env bash

remove_pattern="examples/models/"

# Create tracking branches of all branches
for remote in `git branch -r | grep -v /HEAD`; do git checkout --track $remote ; done

# Remove DIRECTORY_NAME from all commits, then remove the refs to the old commits
# (repeat these two commands for as many directories that you want to remove)
git filter-branch --index-filter "git rm -rf --cached --ignore-unmatch ${remove_pattern}" --prune-empty --tag-name-filter cat -- --all
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Ensure all old refs are fully removed
rm -Rf .git/logs .git/refs/original

# Perform a garbage collection to remove commits with no refs
git gc --prune=all --aggressive

# Force push all branches to overwrite their history
# (use with caution!)
git push origin --all --force
git push origin --tags --force

