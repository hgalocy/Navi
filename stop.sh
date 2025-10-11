#!/usr/bin/env bash
tmux kill-session -t navi 2>/dev/null || true
echo "Navi stack stopped."