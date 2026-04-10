#!/usr/bin/env python
"""
鲁棒性测试结果汇总与 RDRR 计算脚本

功能:
1. 从测试日志中提取 mAP、NDS 等指标
2. 追加到 CSV 文件
3. 计算 RDRR（相对退化减少率）并生成最终报告

使用方式:
    # 提取单次测试结果
    python tools/collect_robust_results.py \
        --test-name drop_frames --test-param ratio10 \
        --log-file robust_results/drop_frames_ratio10/test.log \
        --csv-file robust_results/results_summary.csv
    
    # 计算 RDRR 并汇总
    python tools/collect_robust_results.py \
        --csv-file robust_results/results_summary.csv \
        --compute-rdrr --output robust_results/final_report.txt
"""

import argparse
import csv
import os
import re
import sys


def extract_metrics_from_log(log_file):
    """从测试日志中提取评测指标。"""
    metrics = {}
    
    if not os.path.exists(log_file):
        print(f'[WARN] Log file not found: {log_file}')
        return metrics
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 匹配 nuScenes 官方评测输出格式
    # 格式类似: pts_bbox_NuScenes/NDS: 0.5264
    patterns = {
        'NDS': r"NDS['\"]?\s*[\]:=]\s*([\d.]+)",
        'mAP': r"mAP['\"]?\s*[\]:=]\s*([\d.]+)",
        'mATE': r"mATE['\"]?\s*[\]:=]\s*([\d.]+)",
        'mASE': r"mASE['\"]?\s*[\]:=]\s*([\d.]+)",
        'mAOE': r"mAOE['\"]?\s*[\]:=]\s*([\d.]+)",
        'mAVE': r"mAVE['\"]?\s*[\]:=]\s*([\d.]+)",
        'mAAE': r"mAAE['\"]?\s*[\]:=]\s*([\d.]+)",
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            # 取最后一个匹配（最终结果）
            metrics[key] = float(matches[-1])
    
    # 也尝试从 dict 格式中提取
    # {'pts_bbox_NuScenes/NDS': 0.5264, 'pts_bbox_NuScenes/mAP': 0.4313, ...}
    dict_pattern = r"'[^']*/(NDS|mAP|mATE|mASE|mAOE|mAVE|mAAE)':\s*([\d.]+)"
    dict_matches = re.findall(dict_pattern, content)
    for key, val in dict_matches:
        metrics[key] = float(val)
    
    return metrics


def append_to_csv(csv_file, test_name, test_param, metrics):
    """将测试结果追加到 CSV 文件。"""
    row = [
        test_name,
        test_param,
        metrics.get('mAP', ''),
        metrics.get('NDS', ''),
        metrics.get('mATE', ''),
        metrics.get('mASE', ''),
        metrics.get('mAOE', ''),
        metrics.get('mAVE', ''),
        metrics.get('mAAE', ''),
    ]
    
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['test_type', 'test_param', 'mAP', 'NDS', 
                           'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'])
        writer.writerow(row)
    
    print(f'[OK] Appended {test_name}/{test_param}: mAP={metrics.get("mAP", "N/A")}, NDS={metrics.get("NDS", "N/A")}')


def compute_rdrr(csv_file, output_file):
    """计算 RDRR（相对退化减少率）并生成报告。
    
    RDRR = 1 - (baseline_perf - method_perf) / baseline_perf * 100%
    正值表示退化幅度小于基线。
    """
    # 读取 CSV
    results = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['test_type']}_{row['test_param']}"
            results[key] = row
    
    # 找到基线
    baseline = results.get('baseline_clean', None)
    if not baseline:
        print('[ERROR] No baseline result found!')
        return
    
    baseline_nds = float(baseline['NDS']) if baseline['NDS'] else None
    baseline_map = float(baseline['mAP']) if baseline['mAP'] else None
    
    if not baseline_nds or not baseline_map:
        print('[ERROR] Baseline NDS or mAP is missing!')
        return
    
    lines = []
    lines.append('=' * 80)
    lines.append('BEVFormerV2 鲁棒性测试报告')
    lines.append('=' * 80)
    lines.append(f'\n基线性能: mAP={baseline_map:.4f}, NDS={baseline_nds:.4f}')
    lines.append('')
    
    # 按测试类型分组
    test_groups = {}
    for key, row in results.items():
        if key == 'baseline_clean':
            continue
        test_type = row['test_type']
        if test_type not in test_groups:
            test_groups[test_type] = []
        test_groups[test_type].append(row)
    
    for test_type, rows in sorted(test_groups.items()):
        lines.append(f'\n{"─" * 60}')
        lines.append(f'  {test_type}')
        lines.append(f'{"─" * 60}')
        lines.append(f'{"参数":<20} {"mAP":<10} {"NDS":<10} {"mAP_RDRR":<12} {"NDS_RDRR":<12}')
        lines.append('-' * 60)
        
        for row in sorted(rows, key=lambda x: x['test_param']):
            param = row['test_param']
            try:
                nds = float(row['NDS']) if row['NDS'] else None
                map_val = float(row['mAP']) if row['mAP'] else None
            except (ValueError, TypeError):
                nds = None
                map_val = None
            
            if nds is not None and map_val is not None:
                # 退化幅度 = baseline - current
                nds_degrade = baseline_nds - nds
                map_degrade = baseline_map - map_val
                
                # RDRR 公式：这里直接展示退化幅度的百分比
                # 如果有另一个方法的基线，可以计算相对减少率
                # 这里先展示绝对退化和退化率
                nds_degrade_pct = (nds_degrade / baseline_nds) * 100 if baseline_nds > 0 else 0
                map_degrade_pct = (map_degrade / baseline_map) * 100 if baseline_map > 0 else 0
                
                lines.append(f'{param:<20} {map_val:<10.4f} {nds:<10.4f} {map_degrade_pct:<12.2f}% {nds_degrade_pct:<12.2f}%')
            else:
                lines.append(f'{param:<20} {"N/A":<10} {"N/A":<10} {"N/A":<12} {"N/A":<12}')
    
    lines.append('')
    lines.append('注: RDRR 列显示的是退化百分比（正值表示性能下降百分比）')
    lines.append('    如需计算两个方法之间的相对退化减少率，请使用公式:')
    lines.append('    RDRR = (Δ_baseline - Δ_ours) / Δ_baseline × 100%')
    
    report = '\n'.join(lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)


def main():
    parser = argparse.ArgumentParser(description='收集鲁棒性测试结果')
    parser.add_argument('--test-name', type=str, default='', help='测试名称')
    parser.add_argument('--test-param', type=str, default='', help='测试参数')
    parser.add_argument('--log-file', type=str, default='', help='测试日志文件路径')
    parser.add_argument('--csv-file', type=str, required=True, help='结果 CSV 文件路径')
    parser.add_argument('--compute-rdrr', action='store_true', help='计算 RDRR 并生成报告')
    parser.add_argument('--output', type=str, default='', help='报告输出路径')
    args = parser.parse_args()
    
    if args.compute_rdrr:
        output = args.output or args.csv_file.replace('.csv', '_report.txt')
        compute_rdrr(args.csv_file, output)
    elif args.log_file and args.test_name:
        metrics = extract_metrics_from_log(args.log_file)
        if metrics:
            append_to_csv(args.csv_file, args.test_name, args.test_param, metrics)
        else:
            print(f'[WARN] No metrics found in {args.log_file}')
    else:
        print('Please specify --log-file with --test-name, or --compute-rdrr')
        sys.exit(1)


if __name__ == '__main__':
    main()
