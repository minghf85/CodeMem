#!/bin/bash
# Monitor atommem evaluation progress

EVAL_DIR="/root/autodl-tmp/EvoAtomMem/data/eval_runs"
LOG_FILE="/tmp/atommem_full.log"

echo "=== AtomMem Evaluation Monitor ==="
echo ""

# Check if process is running
if pgrep -f "eval_atommem.py --experiment atommem_full" > /dev/null; then
    echo "✅ Evaluation is RUNNING"
else
    echo "❌ Evaluation is NOT running"
fi

echo ""
echo "--- Latest Log ---"
tail -5 "$LOG_FILE" 2>/dev/null || echo "No log file yet"

echo ""
echo "--- Results Progress ---"
LATEST_RESULT=$(find "$EVAL_DIR" -name "eval_atommem.jsonl" -path "*/atommem_full_*" 2>/dev/null | sort | tail -1)
if [ -f "$LATEST_RESULT" ]; then
    COMPLETED=$(wc -l < "$LATEST_RESULT")
    echo "Completed: $COMPLETED / 1986 QA pairs"
    echo "Progress: $(python3 -c "print(f'{$COMPLETED/1986*100:.1f}%')")"
    echo "Result file: $LATEST_RESULT"

    if [ $COMPLETED -gt 0 ]; then
        echo ""
        echo "--- Current Metrics (from last result) ---"
        tail -1 "$LATEST_RESULT" | python3 -c "import sys, json; r=json.load(sys.stdin); print(f\"msgmem recall: {r.get('msgmem_evidence_recall', 0):.3f}\"); print(f\"atommem recall: {r.get('atommem_evidence_recall', 0):.3f}\")" 2>/dev/null || echo "Cannot parse result"
    fi
else
    echo "No results yet"
fi

echo ""
echo "--- To check full log ---"
echo "tail -f $LOG_FILE"
