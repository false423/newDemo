set -u
set -o pipefail

# ------------------------------------------------------------------------------
# User configuration
# ------------------------------------------------------------------------------
PYTHON_FILE="${PYTHON_FILE:-OUR-Y.py}"
GPU="${GPU:-1}"
SOURCE="${SOURCE:-0}"
N_RUN="${N_RUN:-5}"
EPOCHS="${EPOCHS:-2000}"
SET="${SET:-core}"

# Current target set. Edit if needed.
TARGETS=(
    "Salinas"
    "NC12"
    "WHU-Hi-LongKou"
)

# Optional conda environment.
# Set USE_CONDA=0 if the shell is already inside the correct environment.
USE_CONDA="${USE_CONDA:-1}"
CONDA_ENV="${CONDA_ENV:-Pytorch}"
CONDA_SH="${CONDA_SH:-/home/t1/anaconda3/etc/profile.d/conda.sh}"

# Output
LOG_ROOT="${LOG_ROOT:-./logs_v9_ablation}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-./batch_results_v9}"

mkdir -p "${LOG_ROOT}"
mkdir -p "${ARCHIVE_ROOT}"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
if [[ "${USE_CONDA}" == "1" ]]; then
    if [[ -f "${CONDA_SH}" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV}"
    else
        echo "[WARN] Conda init file not found: ${CONDA_SH}"
        echo "[WARN] Continuing with the current Python environment."
    fi
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${PYTHON_FILE}" ]]; then
    echo "[ERROR] Python file not found: ${PYTHON_FILE}"
    exit 1
fi

# ------------------------------------------------------------------------------
# Experiment definitions
#
# EXP_NAMES:
#   unique human-readable experiment name
#
# INTERNAL_ABLATIONS:
#   value actually passed to --sm_ablation.
#   Needed because the Python code saves by cfg.ablation only.
#
# EXTRA_ARGS:
#   additional switches/lambda overrides
# ------------------------------------------------------------------------------

CORE_NAMES=(
    "full"
    "no_dictionary"
    "no_supcon"
    "random_target_init"
    "no_target_tune"
    "no_target_anchor"
    "ce_only"
)

CORE_INTERNAL=(
    "full"
    "no_dictionary"
    "no_supcon"
    "random_target_init"
    "no_target_tune"
    "full"
    "no_dictionary"
)

CORE_ARGS=(
    ""
    ""
    ""
    ""
    ""
    "--sm_lambda_target_anchor 0"
    "--sm_lambda_supcon 0"
)

# Fine-grained dictionary regularizer ablations.
EXT_NAMES=(
    "no_dict_diversity"
    "no_dict_selectivity"
    "no_dict_balance"
    "no_dict_regularization"
)

EXT_INTERNAL=(
    "full"
    "full"
    "full"
    "full"
)

EXT_ARGS=(
    "--sm_lambda_dict_diversity 0"
    "--sm_lambda_dict_selectivity 0"
    "--sm_lambda_dict_balance 0"
    "--sm_lambda_dict_diversity 0 --sm_lambda_dict_selectivity 0 --sm_lambda_dict_balance 0"
)

EXP_NAMES=("${CORE_NAMES[@]}")
INTERNAL_ABLATIONS=("${CORE_INTERNAL[@]}")
EXTRA_ARGS=("${CORE_ARGS[@]}")

if [[ "${SET}" == "extended" ]]; then
    EXP_NAMES+=("${EXT_NAMES[@]}")
    INTERNAL_ABLATIONS+=("${EXT_INTERNAL[@]}")
    EXTRA_ARGS+=("${EXT_ARGS[@]}")
elif [[ "${SET}" != "core" ]]; then
    echo "[ERROR] SET must be core or extended. Got: ${SET}"
    exit 1
fi

# ------------------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------------------
MANIFEST="${ARCHIVE_ROOT}/manifest.txt"

{
    echo "================================================================================"
    echo "OUR-V9 Semantic Manifold Ablation Batch"
    echo "Start: $(date)"
    echo "Python: ${PYTHON_FILE}"
    echo "GPU: ${GPU}"
    echo "SOURCE: ${SOURCE}"
    echo "N_RUN: ${N_RUN}"
    echo "EPOCHS: ${EPOCHS}"
    echo "SET: ${SET}"
    echo "Targets: ${TARGETS[*]}"
    echo "Experiments: ${EXP_NAMES[*]}"
    echo "================================================================================"
} | tee "${MANIFEST}"

TOTAL=$(( ${#TARGETS[@]} * ${#EXP_NAMES[@]} ))
COUNT=0
FAILED=0

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
for TARGET in "${TARGETS[@]}"; do

    for IDX in "${!EXP_NAMES[@]}"; do
        EXP="${EXP_NAMES[$IDX]}"
        INTERNAL="${INTERNAL_ABLATIONS[$IDX]}"
        EXTRA="${EXTRA_ARGS[$IDX]}"

        COUNT=$((COUNT + 1))

        TARGET_LOG_DIR="${LOG_ROOT}/${TARGET}"
        mkdir -p "${TARGET_LOG_DIR}"

        LOG_FILE="${TARGET_LOG_DIR}/${EXP}.log"
        DEST_DIR="${ARCHIVE_ROOT}/${TARGET}/${EXP}"

        echo ""
        echo "################################################################################"
        echo "# [${COUNT}/${TOTAL}] Target=${TARGET} | Experiment=${EXP}"
        echo "# Internal --sm_ablation=${INTERNAL}"
        echo "# Extra args: ${EXTRA:-<none>}"
        echo "# Start: $(date)"
        echo "# Log: ${LOG_FILE}"
        echo "################################################################################"

        # shellcheck disable=SC2086
        python "${PYTHON_FILE}" \
            --source "${SOURCE}" \
            --target "${TARGET}" \
            --n_run "${N_RUN}" \
            --epochs "${EPOCHS}" \
            --sm_ablation "${INTERNAL}" \
            ${EXTRA} \
            2>&1 | tee "${LOG_FILE}"

        RC=${PIPESTATUS[0]}

        if [[ "${RC}" -ne 0 ]]; then
            echo "[FAILED] Target=${TARGET} | Experiment=${EXP} | exit=${RC}" \
                | tee -a "${MANIFEST}"
            FAILED=$((FAILED + 1))
            continue
        fi

        # ----------------------------------------------------------------------
        # Archive the Python output BEFORE the next lambda-based variant can
        # overwrite the same cfg.ablation directory.
        # ----------------------------------------------------------------------
        INTERNAL_DIR="./checkpoints_v9_semantic_manifold/${TARGET}/${INTERNAL}"

        rm -rf "${DEST_DIR}"
        mkdir -p "${DEST_DIR}"

        if [[ -d "${INTERNAL_DIR}" ]]; then
            cp -a "${INTERNAL_DIR}/." "${DEST_DIR}/"
        else
            echo "[WARN] Python result directory not found: ${INTERNAL_DIR}"
        fi

        cp "${LOG_FILE}" "${DEST_DIR}/run.log"

        # Save exact command metadata.
        {
            echo "experiment_name=${EXP}"
            echo "internal_ablation=${INTERNAL}"
            echo "extra_args=${EXTRA}"
            echo "target=${TARGET}"
            echo "source=${SOURCE}"
            echo "n_run=${N_RUN}"
            echo "epochs=${EPOCHS}"
            echo "gpu=${GPU}"
            echo "finished=$(date)"
        } > "${DEST_DIR}/experiment_meta.txt"

        echo "[DONE] Target=${TARGET} | Experiment=${EXP}" \
            | tee -a "${MANIFEST}"
    done
done

# ------------------------------------------------------------------------------
# Collect a compact CSV from archived per_run_metrics.csv files.
# This does NOT require pandas.
# ------------------------------------------------------------------------------
SUMMARY_CSV="${ARCHIVE_ROOT}/ablation_summary.csv"

python - "${ARCHIVE_ROOT}" "${SUMMARY_CSV}" <<'PY'
import csv
import math
import os
import statistics
import sys

root = sys.argv[1]
out = sys.argv[2]

rows = []

for target in sorted(os.listdir(root)):
    target_dir = os.path.join(root, target)
    if not os.path.isdir(target_dir):
        continue

    for exp in sorted(os.listdir(target_dir)):
        exp_dir = os.path.join(target_dir, exp)
        metrics = os.path.join(exp_dir, "per_run_metrics.csv")

        if not os.path.isfile(metrics):
            continue

        with open(metrics, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)

        if not data:
            continue

        def vals(key):
            out_vals = []
            for r in data:
                try:
                    v = float(r[key])
                except Exception:
                    continue
                if math.isfinite(v):
                    out_vals.append(v)
            return out_vals

        oa = vals("oa")
        aa = vals("aa")
        kp = vals("kappa")

        def mean_std(v):
            if not v:
                return float("nan"), float("nan")
            m = statistics.mean(v)
            s = statistics.pstdev(v) if len(v) > 1 else 0.0
            return m, s

        oa_m, oa_s = mean_std(oa)
        aa_m, aa_s = mean_std(aa)
        kp_m, kp_s = mean_std(kp)

        rows.append({
            "target": target,
            "experiment": exp,
            "n_runs": len(data),
            "oa_mean_percent": 100.0 * oa_m,
            "oa_std_percent": 100.0 * oa_s,
            "aa_mean_percent": 100.0 * aa_m,
            "aa_std_percent": 100.0 * aa_s,
            "kappa_mean": kp_m,
            "kappa_std": kp_s,
        })

fields = [
    "target",
    "experiment",
    "n_runs",
    "oa_mean_percent",
    "oa_std_percent",
    "aa_mean_percent",
    "aa_std_percent",
    "kappa_mean",
    "kappa_std",
]

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"[SUMMARY] {out}")
PY

echo ""
echo "================================================================================"
echo "Batch finished: $(date)"
echo "Total experiments: ${TOTAL}"
echo "Failed: ${FAILED}"
echo "Archived results: ${ARCHIVE_ROOT}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "================================================================================"

exit "${FAILED}"