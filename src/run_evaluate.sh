#!/bin/bash
conda activate marlin23
python experiment_manager.py sac schema_update
# Locate the current project from the script location.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SRC_DIR/.." && pwd)"

AGENT_DIR="$PROJECT_ROOT/results/agents"
MODEL_PATH="$AGENT_DIR/sac_agent_final.pkl"
RESULT_DIR="$PROJECT_ROOT/results"

echo "=============================================="
echo "Evaluation configuration"
echo "=============================================="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Source directory:  $SRC_DIR"
echo "Project root:      $PROJECT_ROOT"
echo "Agent directory:   $AGENT_DIR"
echo "Model path:        $MODEL_PATH"
echo "Result directory:  $RESULT_DIR"
echo "=============================================="

if [ ! -d "$AGENT_DIR" ]; then
    echo "ERROR: Agent directory does not exist:"
    echo "$AGENT_DIR"
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Final trained agent was not found:"
    echo "$MODEL_PATH"
    echo ""
    echo "Available agent files:"
    ls -l "$AGENT_DIR"
    return 1 2>/dev/null || exit 1
fi

cd "$SRC_DIR" || {
    echo "ERROR: Could not enter source directory:"
    echo "$SRC_DIR"
    return 1 2>/dev/null || exit 1
}

python evaluate.py \
    --model_path "$MODEL_PATH" \
    --simulation_id "1_Building_1" \
    --root_dir "$RESULT_DIR" \
    --deterministic

EVALUATION_STATUS=$?

if [ "$EVALUATION_STATUS" -ne 0 ]; then
    echo ""
    echo "Evaluation failed with exit code: $EVALUATION_STATUS"
    return "$EVALUATION_STATUS" 2>/dev/null || exit "$EVALUATION_STATUS"
fi

echo ""
echo "Evaluation completed successfully."