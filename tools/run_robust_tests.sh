#!/bin/bash
# ============================================================================
# BEVFormerV2 三大鲁棒性测试 —— 一键批量测试脚本
# 
# 使用方式:
#   bash tools/run_robust_tests.sh <CONFIG> <CHECKPOINT> <GPUS> [OUTPUT_DIR]
#
# 示例:
#   bash tools/run_robust_tests.sh \
#       projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py \
#       ckpts/bevformerv2-r50-t2-48ep.pth \
#       4 \
#       robust_results
# ============================================================================

set -e

CONFIG=${1:-"projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py"}
CHECKPOINT=${2:-"ckpts/bevformerv2-r50-t2-48ep.pth"}
GPUS=${3:-4}
OUTPUT_DIR=${4:-"robust_results"}
PORT=${PORT:-29503}

# noise pkl 路径
NOISE_PKL="data/nuscenes_infos_val_with_noise.pkl"      # 外参扰动 + 遮挡 mask
DROP_PKL="data/nuscenes_infos_val_with_noise_Drop .pkl"  # 丢帧数据（注意文件名有空格）
MASK_DIR="robust_benchmark/Occlusion_mask"

# 基线使用的 ann_file（如果没有原始 temporal_val.pkl，则用 noise pkl 替代）
BASELINE_ANN_FILE="${NOISE_PKL}"

mkdir -p ${OUTPUT_DIR}

# 日志文件
LOG_FILE="${OUTPUT_DIR}/test_log_$(date +%Y%m%d_%H%M%S).log"
RESULTS_CSV="${OUTPUT_DIR}/results_summary.csv"

# CSV 表头
echo "test_type,test_param,mAP,NDS,mATE,mASE,mAOE,mAVE,mAAE" > ${RESULTS_CSV}

# ============================================================================
# 辅助函数: 运行单次测试
# ============================================================================
run_single_test() {
    local TEST_NAME=$1
    local TEST_PARAM=$2
    shift 2
    # 剩余参数作为 cfg-options 的 key=value 对
    local CFG_ARGS=("$@")
    local RESULT_DIR="${OUTPUT_DIR}/${TEST_NAME}_${TEST_PARAM}"
    
    mkdir -p "${RESULT_DIR}"

    echo ""
    echo "================================================================"
    echo "[$(date)] Running: ${TEST_NAME} - ${TEST_PARAM}"
    echo "  CFG_ARGS: ${CFG_ARGS[*]}"
    echo "================================================================"
    
    # 构建命令
    local CMD_ARGS=(
        python -m torch.distributed.launch
        --nproc_per_node=$GPUS
        --master_port=$PORT
        "$(dirname "$0")/test.py"
        "${CONFIG}" "${CHECKPOINT}"
        --launcher pytorch
        --eval bbox
    )
    
    # 如果有 cfg-options 参数，添加 --cfg-options
    if [ ${#CFG_ARGS[@]} -gt 0 ]; then
        CMD_ARGS+=(--cfg-options)
        CMD_ARGS+=("${CFG_ARGS[@]}")
    fi
    
    # 运行测试
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    "${CMD_ARGS[@]}" 2>&1 | tee "${RESULT_DIR}/test.log"
    
    # 递增端口避免冲突
    PORT=$((PORT + 1))
    
    # 提取结果
    python tools/collect_robust_results.py \
        --test-name "${TEST_NAME}" \
        --test-param "${TEST_PARAM}" \
        --log-file "${RESULT_DIR}/test.log" \
        --csv-file "${RESULTS_CSV}" \
        2>&1 || echo "[WARN] Failed to collect results for ${TEST_NAME}_${TEST_PARAM}"
    
    echo "[$(date)] Completed: ${TEST_NAME} - ${TEST_PARAM}"
}


# ============================================================================
# 0. 基线测试 (无噪声)
# ============================================================================
echo "===== [Phase 0] Baseline Test ====="
# 基线测试：覆盖 ann_file 指向已有的 pkl 文件
run_single_test "baseline" "clean" \
    "data.test.ann_file=${BASELINE_ANN_FILE}"


# ============================================================================
# 1. 丢帧测试 (10%~90%)
# ============================================================================
echo ""
echo "===== [Phase 1] Frame Drop Tests ====="

for RATIO in 10 20 30 40 50 60 70 80 90; do
    run_single_test "drop_frames" "ratio${RATIO}" \
        "data.test.type=NuScenesNoiseDatasetV2" \
        "data.test.noise_nuscenes_ann_file=${DROP_PKL}" \
        "data.test.drop_frames=True" \
        "data.test.drop_ratio=${RATIO}" \
        "data.test.drop_type=discrete"
done


# ============================================================================
# 2. 外参扰动测试
# ============================================================================
echo ""
echo "===== [Phase 2] Extrinsics Perturbation Tests ====="

# 2a. 单摄像头扰动
for LEVEL in L1 L2 L3 L4; do
    run_single_test "extrinsics_single" "${LEVEL}" \
        "data.test.type=NuScenesNoiseDatasetV2" \
        "data.test.noise_nuscenes_ann_file=${NOISE_PKL}" \
        "data.test.extrinsics_noise=True" \
        "data.test.extrinsics_noise_level=${LEVEL}" \
        "data.test.extrinsics_noise_scope=single"
done

# 2b. 多摄像头扰动
for LEVEL in L1 L2 L3 L4; do
    run_single_test "extrinsics_all" "${LEVEL}" \
        "data.test.type=NuScenesNoiseDatasetV2" \
        "data.test.noise_nuscenes_ann_file=${NOISE_PKL}" \
        "data.test.extrinsics_noise=True" \
        "data.test.extrinsics_noise_level=${LEVEL}" \
        "data.test.extrinsics_noise_scope=all"
done


# ============================================================================
# 3. 遮挡测试 (S1~S4)
# ============================================================================
echo ""
echo "===== [Phase 3] Occlusion Tests ====="

# S1: exp=1.0, S2: exp=2.0, S3: exp=3.0, S4: exp=5.0
declare -A OCCLUSION_EXPS
OCCLUSION_EXPS=( ["S1"]="1.0" ["S2"]="2.0" ["S3"]="3.0" ["S4"]="5.0" )

for LEVEL in S1 S2 S3 S4; do
    EXP=${OCCLUSION_EXPS[$LEVEL]}
    
    # 遮挡测试需要替换 Pipeline 中的 LoadMultiViewImageFromFiles
    run_single_test "occlusion" "${LEVEL}_exp${EXP}" \
        "data.test.pipeline.0.type=LoadMaskMultiViewImageFromFiles" \
        "data.test.pipeline.0.noise_nuscenes_ann_file=${NOISE_PKL}" \
        "data.test.pipeline.0.mask_file=${MASK_DIR}" \
        "data.test.pipeline.0.mask_exp=${EXP}"
done


# ============================================================================
# 4. 结果汇总
# ============================================================================
echo ""
echo "================================================================"
echo "[$(date)] All tests completed!"
echo "================================================================"
echo ""
echo "Results saved to: ${RESULTS_CSV}"
echo ""

# 计算 RDRR 并生成最终报告
python tools/collect_robust_results.py \
    --csv-file "${RESULTS_CSV}" \
    --compute-rdrr \
    --output "${OUTPUT_DIR}/final_report.txt" \
    2>&1

echo "Final report: ${OUTPUT_DIR}/final_report.txt"
cat "${OUTPUT_DIR}/final_report.txt"
