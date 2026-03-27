#!/bin/bash
JOBS=(22565213 22565214 22565164 22565038 22565039)
NAMES=(eval_favc_B1 A3 B2 B3 B_all)
LOGS=(
  /data/projects/punim2637/nnliang/AVH-Align/logs/eval_favc_B1_22565213.out
  /data/projects/punim2637/nnliang/AVH-Align/logs/train_A3_22565214.out
  /data/projects/punim2637/nnliang/AVH-Align/logs/train_B2_22565164.out
  /data/projects/punim2637/nnliang/AVH-Align/logs/train_B3_22565038.out
  /data/projects/punim2637/nnliang/AVH-Align/logs/train_B_all_22565039.out
)

while true; do
  echo ""
  echo "════════════════════════  $(date '+%Y-%m-%d %H:%M:%S')  ════════════════════════"
  squeue -u "$USER" 2>/dev/null
  echo ""

  for i in 0 1 2 3 4; do
    JID="${JOBS[$i]}"
    NAME="${NAMES[$i]}"
    LOG="${LOGS[$i]}"
    if [[ -f "$LOG" ]]; then
      LAST=$(grep "AUC causal=" "$LOG" 2>/dev/null | tail -3)
      if [[ -n "$LAST" ]]; then
        echo "  [$NAME]"
        echo "$LAST" | sed 's/^/    /'
      else
        echo "  [$NAME] log exists — waiting for first epoch..."
      fi
    else
      STATE=$(squeue -j "$JID" -h -o "%T" 2>/dev/null)
      echo "  [$NAME] ${STATE:-PENDING (no log yet)}"
    fi
  done

  # Check if all finished
  ALL_DONE=true
  for JID in "${JOBS[@]}"; do
    if squeue -j "$JID" -h -o "%T" 2>/dev/null | grep -q .; then
      ALL_DONE=false
    fi
  done
  if $ALL_DONE; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━  ALL JOBS FINISHED  ━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    for i in 0 1 2 3 4; do
      NAME="${NAMES[$i]}"
      LOG="${LOGS[$i]}"
      echo "══ $NAME final AUC history ══"
      grep "AUC causal=" "$LOG" 2>/dev/null || echo "  (log not found)"
      echo ""
    done
    break
  fi

  sleep 120
done
