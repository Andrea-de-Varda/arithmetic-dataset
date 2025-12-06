#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --output=eval_arith_%j.out

set -euo pipefail

# ============================================
# ARGUMENT PARSING
# ============================================
MODEL=""
REASONING=""
DATASET="all"
SPLIT="test"
MAX_SAMPLES=""
DTYPE="auto"

usage() {
    echo "Usage: sbatch eval_models.sh --modelname <model> [options]"
    echo ""
    echo "Required:"
    echo "  --modelname NAME    HuggingFace model name (e.g., mistralai/Mistral-7B-v0.1)"
    echo ""
    echo "Optional:"
    echo "  --reasoning         Use reasoning mode (for reasoning LLMs like Magistral)"
    echo "  --dataset NAME      Dataset to evaluate: all|numeric|english|spanish|italian|embedded (default: all)"
    echo "  --split NAME        Split to evaluate: train|val|test (default: test)"
    echo "  --max-samples N     Limit evaluation to N samples (for testing)"
    echo "  --dtype TYPE        Model dtype: auto|float16|bfloat16|float32 (default: auto)"
    echo ""
    echo "Examples:"
    echo "  sbatch eval_models.sh --modelname mistralai/Mistral-7B-v0.1"
    echo "  sbatch eval_models.sh --modelname mistralai/Magistral-Small-2509 --reasoning"
    echo "  sbatch eval_models.sh --modelname meta-llama/Llama-2-7b-hf --dataset numeric --max-samples 100"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --modelname)
            MODEL="$2"
            shift 2
            ;;
        --reasoning)
            REASONING="--reasoning"
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: --modelname is required"
    usage
fi

# ============================================
# PRINT CONFIGURATION
# ============================================
echo "============================================"
echo "Arithmetic Dataset Evaluation"
echo "============================================"
echo "Model:     $MODEL"
echo "Mode:      ${REASONING:-standard}"
echo "Dataset:   $DATASET"
echo "Split:     $SPLIT"
echo "Dtype:     $DTYPE"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: ${MAX_SAMPLES#--max-samples }"
fi
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "============================================"
echo ""

# ============================================
# ENVIRONMENT SETUP
# ============================================
source /etc/profile.d/modules.sh

module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate modularity-conda
conda deactivate
conda activate modularity-conda

# Install/upgrade required packages
pip install --upgrade accelerate transformers protobuf tqdm -q

# HuggingFace cache configuration
export HF_HOME=/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# Offload directory for large models
export HF_OFFLOAD_DIR=/orcd/data/evelina9/001/USERS/devar_ag/offload
mkdir -p "$HF_OFFLOAD_DIR"

# ============================================
# PROJECT SETUP
# ============================================
# MODIFY THIS PATH TO YOUR PROJECT DIRECTORY
PROJECT_DIR=/orcd/data/evelina9/001/USERS/devar_ag/arithmetic-dataset

cd "$PROJECT_DIR"

# Create results directory
mkdir -p results

# ============================================
# RUN EVALUATION
# ============================================
echo "Starting evaluation..."
echo ""

python eval_models.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --dtype "$DTYPE" \
    --output-dir results \
    $REASONING \
    $MAX_SAMPLES

echo ""
echo "============================================"
echo "Evaluation Complete"
echo "============================================"

