# 实验数据汇总

- 生成时间: 2026-04-02 01:19:02 AEDT
- 训练输出目录数: 67
- 含训练日志的实验数: 65
- 含评测结果的实验目录数: 10
- 找到的 eval_results.txt 总数: 23

## 训练阶段 Top 10（按 best_val_auc_causal）

| 实验 | best_val_auc_causal | best_val_auc_full | max_epoch | 评测文件数 | 最优指标文件 |
|---|---:|---:|---:|---:|---|
| outputs_A34_e2e | 1.0000 | - | 17 | 0 | outputs_A34_e2e/logs/version_0/metrics.csv |
| outputs_A33_e2e | 1.0000 | - | 25 | 0 | outputs_A33_e2e/logs/version_0/metrics.csv |
| outputs_A39_e2e | 1.0000 | - | 16 | 0 | outputs_A39_e2e/logs/version_1/metrics.csv |
| outputs_A6_e2e | 1.0000 | - | 21 | 0 | outputs_A6_e2e/logs/version_2/metrics.csv |
| outputs_A34_v2_e2e | 1.0000 | - | 13 | 0 | outputs_A34_v2_e2e/logs/version_0/metrics.csv |
| outputs_A1_e2e_dadv | 1.0000 | - | 29 | 0 | outputs_A1_e2e_dadv/logs/version_3/metrics.csv |
| outputs_A28_e2e | 1.0000 | - | 4 | 0 | outputs_A28_e2e/logs/version_0/metrics.csv |
| outputs_A15_e2e | 1.0000 | - | 2 | 0 | outputs_A15_e2e/logs/version_1/metrics.csv |
| outputs_A9_e2e | 1.0000 | - | 18 | 0 | outputs_A9_e2e/logs/version_0/metrics.csv |
| outputs_A32_e2e | 1.0000 | - | 2 | 0 | outputs_A32_e2e/logs/version_1/metrics.csv |

## 可比测试集结果 Top（优先看 causal 头，其次 single/full）

### full, N=1114

| 实验 | 模型 | 使用头 | overall AUC | overall AP | eval 路径 |
|---|---|---|---:|---:|---|
| outputs_A6 | fcd_a6 | causal | 0.8885 | 0.9971 | outputs_A6/results/eval_results.txt |
| outputs_A7 | fcd_a7 | causal | 0.8251 | 0.9953 | outputs_A7/results/eval_results.txt |
| outputs_A5 | fcd | causal | 0.8186 | 0.9948 | outputs_A5/results/eval_results.txt |
| outputs_A8 | sad_a8 | causal | 0.7729 | 0.9937 | outputs_A8/results/eval_results.txt |
| results | baseline | single | 0.7684 | 0.9935 | results/favc_baseline/eval_results.txt |
| outputs_causal_mi | causal_mi | causal | 0.7661 | 0.9934 | outputs_causal_mi/results/eval_results.txt |
| results | causal | causal | 0.7598 | 0.9933 | results/favc_causal/eval_results.txt |
| outputs_A3 | ablation | causal | 0.7490 | 0.9930 | outputs_A3/results/eval_results.txt |

### full, N=4089

| 实验 | 模型 | 使用头 | overall AUC | overall AP | eval 路径 |
|---|---|---|---:|---:|---|
| avh_sup | ablation | causal | 0.7806 | 0.9936 | avh_sup/outputs_ablation_a1/results_favc/eval_results.txt |
| outputs_A1 | ablation | causal | 0.7806 | 0.9936 | outputs_A1/results_favc/eval_results.txt |
| avh_sup | ablation | causal | 0.7648 | 0.9931 | avh_sup/outputs_A2/results_favc/eval_results.txt |
| outputs_B1 | modal | causal | 0.7538 | 0.9927 | outputs_B1/results/favc_eval/eval_results.txt |
| avh_sup | causal | causal | 0.7397 | 0.9923 | avh_sup/outputs_causal/results_favc/eval_results.txt |
| avh_sup | ablation | causal | 0.7275 | 0.9919 | avh_sup/outputs_A1/results_favc/eval_results.txt |
| avh_sup | ablation | causal | 0.2657 | 0.9547 | avh_sup/outputs_A0/results_favc/eval_results.txt |

### trimmed, N=1114

| 实验 | 模型 | 使用头 | overall AUC | overall AP | eval 路径 |
|---|---|---|---:|---:|---|
| outputs_A6 | fcd_a6 | causal | 0.8187 | 0.9949 | outputs_A6/results/trimmed/eval_results.txt |
| outputs_A7 | fcd_a7 | causal | 0.7778 | 0.9937 | outputs_A7/results/trimmed/eval_results.txt |
| outputs_A5 | fcd | causal | 0.7489 | 0.9910 | outputs_A5/results/trimmed/eval_results.txt |
| outputs_baseline | baseline | single | 0.7183 | 0.9916 | outputs_baseline/results_favc_trimmed/eval_results.txt |
| outputs_A1 | ablation | causal | 0.7083 | 0.9909 | outputs_A1/results_favc_trimmed/eval_results.txt |
| outputs_A8 | sad_a8 | causal | 0.7004 | 0.9906 | outputs_A8/results/trimmed/eval_results.txt |
| outputs_causal | causal | causal | 0.6788 | 0.9902 | outputs_causal/results_favc_trimmed/eval_results.txt |
| outputs_A3 | ablation | causal | 0.6704 | 0.9901 | outputs_A3/results/trimmed/eval_results.txt |

## 多随机种子汇总

| 基础实验 | seeds | mean(best_val_auc_causal) | std |
|---|---:|---:|---:|
| outputs_A0 | 3 | 0.5073 | 0.0039 |
| outputs_A1 | 3 | 0.9982 | 0.0005 |
| outputs_A2 | 3 | 0.9980 | 0.0004 |

## 备注

- 另外检测到 5 个历史评测文件位于 avh_sup/avh_sup/outputs_* 子目录，已保留在 experiment_eval_summary.csv 和 experiment_eval_details.csv 中。
- 训练汇总仅统计 AVH-Align/avh_sup 顶层 outputs* 目录，避免把嵌套旧目录重复算进训练实验数。
