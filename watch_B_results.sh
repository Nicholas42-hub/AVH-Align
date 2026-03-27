#!/bin/bash
# ============================================================
#  Watch modal balance jobs and print results as they finish.
#  Usage: bash watch_B_results.sh
# ============================================================

JOB_IDS=(22565036 22565037 22565038 22565039)
JOB_NAMES=(B1 B2 B3 B_all)
LOG_FILES=(
    logs/train_B1_22565036.out
    logs/train_B2_22565037.out
    logs/train_B3_22565038.out
    logs/train_B_all_22565039.out
)

DONE=()

cd /data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  Watching B1/B2/B3/B_all jobs — $(date)"
echo "  Polling every 60s. Ctrl-C to stop."
echo "============================================================"

while true; do
    ALL_DONE=true
    for i in "${!JOB_IDS[@]}"; do
        JID=${JOB_IDS[$i]}
        NAME=${JOB_NAMES[$i]}

        # Skip if already reported
        if [[ " ${DONE[*]} " == *" $JID "* ]]; then
            continue
        fi

        STATE=$(squeue -j "$JID" -h -o "%T" 2>/dev/null)
        if [[ -z "$STATE" ]]; then
            # Job no longer in queue → finished (or failed)
            DONE+=("$JID")
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  Job ${NAME} (${JID}) finished — $(date)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            # Print last 40 lines of stdout (contains val AUC per epoch + final)
            LOG="${LOG_FILES[$i]}"
            if [[ -f "$LOG" ]]; then
                echo "  [stdout: last 40 lines]"
                tail -40 "$LOG"
            else
                # Glob for the log (SLURM appends job id)
                FOUND=$(ls logs/train_${NAME}_${JID}.out 2>/dev/null | head -1)
                if [[ -n "$FOUND" ]]; then
                    echo "  [stdout: last 40 lines — $FOUND]"
                    tail -40 "$FOUND"
                else
                    echo "  (log file not found: ${LOG})"
                fi
            fi

            # Print eval_results_modal.txt if test was run
            RESULTS="avh_sup/outputs_${NAME}/results/eval_results_modal.txt"
            if [[ -f "$RESULTS" ]]; then
                echo ""
                echo "  [eval_results_modal.txt]"
                cat "$RESULTS"
            fi
        else
            ALL_DONE=false
            echo "  $(date +%H:%M:%S)  ${NAME} (${JID}) — ${STATE}"
        fi
    done

    if $ALL_DONE && [[ ${#DONE[@]} -eq ${#JOB_IDS[@]} ]]; then
        echo ""
        echo "============================================================"
        echo "  All jobs done — $(date)"
        echo "============================================================"
        break
    fi

    sleep 60
done
