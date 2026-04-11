#!/bin/bash
# ============================================================================
# 补跑缺失的鲁棒性测试
# ============================================================================

set -e

CONFIG=${1:-"projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py"}
CHECKPOINT=${2:-"ckpts/epoch_48.pth"}
GPUS=${3:-4}
OUTPUT_DIR=${4:-"robust_results"}
PORT=${PORT:-29600}

# 数据路径
NOISE_PKL="data/nuscenes/nuscenes_infos_val_with_noise.pkl"
DROP_PKL="data/nuscenes/nuscenes_infos_val_with_noise_Drop.pkl"
MASK_DIR="robust_benchmark/Occlusion_mask"
BASELINE_ANN_FILE="data/nuscenes/nuscenes2d_temporal_infos_val.pkl"

RESULTS_CSV="${OUTPUT_DIR}/results_summary.csv"

# ============================================================================
# 辅助函数
# ============================================================================
run_single_test() {
    local TEST_NAME=$1
    local TEST_PARAM=$2
    shift 2
    local CFG_ARGS=("$@")
    local RESULT_DIR="${OUTPUT_DIR}/${TEST_NAME}_${TEST_PARAM}"
    
    mkdir -p "${RESULT_DIR}"

    echo ""
    echo "================================================================"
    echo "[$(date)] Running: ${TEST_NAME} - ${TEST_PARAM}"
    echo "  CFG_ARGS: ${CFG_ARGS[*]}"
    echo "================================================================"
    
    local CMD_ARGS=(
        python -m torch.distributed.launch
        --nproc_per_node=$GPUS
        --master_port=$PORT
        "$(dirname "$0")/test.py"
        "${CONFIG}" "${CHECKPOINT}"
        --launcher pytorch
        --eval bbox
    )
    
    if [ ${#CFG_ARGS[@]} -gt 0 ]; then
        CMD_ARGS+=(--cfg-options)
        CMD_ARGS+=("${CFG_ARGS[@]}")
    fi
    
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    "${CMD_ARGS[@]}" 2>&1 | tee "${RESULT_DIR}/test.log"
    
    PORT=$((PORT + 1))
    
    python tools/collect_robust_results.py \
        --test-name "${TEST_NAME}" \
        --test-param "${TEST_PARAM}" \
        --log-file "${RESULT_DIR}/test.log" \
        --csv-file "${RESULTS_CSV}" \
        2>&1 || echo "[WARN] Failed to collect results for ${TEST_NAME}_${TEST_PARAM}"
    
    echo "[$(date)] Completed: ${TEST_NAME} - ${TEST_PARAM}"
}

# ============================================================================
# 1. 基线测试
# ============================================================================
echo "===== [补跑] Baseline Test ====="
run_single_test "baseline" "clean" \
    "data.test.ann_file=${BASELINE_ANN_FILE}"

# ============================================================================
# 2. 丢帧 ratio80
# ============================================================================
echo "===== [补跑] Drop Frames ratio80 ====="
run_single_test "drop_frames" "ratio80" \
    "data.test.type=NuScenesNoiseDatasetV2" \
    "data.test.ann_file=${BASELINE_ANN_FILE}" \
    "data.test.noise_nuscenes_ann_file=${DROP_PKL}" \
    "data.test.drop_frames=True" \
    "data.test.drop_ratio=80" \
    "data.test.drop_type=discrete"

# ============================================================================
# 3. 外参扰动 - 单摄像头 L2~L4
# ============================================================================
echo "===== [补跑] Extrinsics Single L2~L4 ====="
for LEVEL in L2 L3 L4; do
    run_single_test "extrinsics_single" "${LEVEL}" \
        "data.test.type=NuScenesNoiseDatasetV2" \
        "data.test.ann_file=${BASELINE_ANN_FILE}" \
        "data.test.noise_nuscenes_ann_file=${NOISE_PKL}" \
        "data.test.extrinsics_noise=True" \
        "data.test.extrinsics_noise_level=${LEVEL}" \
        "data.test.extrinsics_noise_scope=single"
done

# ============================================================================
# 4. 外参扰动 - 多摄像头 L1~L4
# ============================================================================
echo "===== [补跑] Extrinsics All L1~L4 ====="
for LEVEL in L1 L2 L3 L4; do
    run_single_test "extrinsics_all" "${LEVEL}" \
        "data.test.type=NuScenesNoiseDatasetV2" \
        "data.test.ann_file=${BASELINE_ANN_FILE}" \
        "data.test.noise_nuscenes_ann_file=${NOISE_PKL}" \
        "data.test.extrinsics_noise=True" \
        "data.test.extrinsics_noise_level=${LEVEL}" \
        "data.test.extrinsics_noise_scope=all"
done

# ============================================================================
# 5. 遮挡测试 S1~S4
# ============================================================================
echo "===== [补跑] Occlusion S1~S4 ====="
declare -A OCCLUSION_EXPS
OCCLUSION_EXPS=( ["S1"]="1.0" ["S2"]="2.0" ["S3"]="3.0" ["S4"]="5.0" )

for LEVEL in S1 S2 S3 S4; do
    EXP=${OCCLUSION_EXPS[$LEVEL]}
    run_single_test "occlusion" "${LEVEL}_exp${EXP}" \
        "data.test.ann_file=${BASELINE_ANN_FILE}" \
        "data.test.pipeline.0.type=LoadMaskMultiViewImageFromFiles" \
        "data.test.pipeline.0.noise_nuscenes_ann_file=${NOISE_PKL}" \
        "data.test.pipeline.0.mask_file=${MASK_DIR}" \
        "data.test.pipeline.0.mask_exp=${EXP}"
done

# ============================================================================
# 6. 生成最终报告
# ============================================================================
echo ""
echo "================================================================"
echo "[$(date)] All remaining tests completed!"
echo "================================================================"

python tools/collect_robust_results.py \
    --csv-file "${RESULTS_CSV}" \
    --compute-rdrr \
    --output "${OUTPUT_DIR}/final_report.txt" \
    2>&1

echo "Final report: ${OUTPUT_DIR}/final_report.txt"
cat "${OUTPUT_DIR}/final_report.txt"
