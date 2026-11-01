#!/bin/bash
# One line per new comment, review comment or review on PR #999 posted after this starts.
cd /Users/moosevan/git/scratchy/.claude/worktrees/spyre-attn-triton
mkdir -p .scratch/pr999_seen
since=${SINCE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}
while true; do
  for ep in issues/999/comments pulls/999/comments pulls/999/reviews; do
    gh api "repos/AI-native-Systems-Research/scratchy/$ep" --paginate \
      --jq '.[] | "\(.id)\t\(.created_at // .submitted_at)\t\(.user.login)\t\(.html_url)"' 2>/dev/null |
      while IFS=$'\t' read -r id at who url; do
        if [[ ! "$at" < "$since" ]] && [ ! -e ".scratch/pr999_seen/$id" ]; then
          : > ".scratch/pr999_seen/$id"
          echo "new $ep by $who at $at: $url"
        fi
      done
  done
  sleep 60
done
