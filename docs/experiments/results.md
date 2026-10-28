# 保存结果看板

由 `python tools/results.py --write` 从公开的逐起点摘要生成。
单位：MACE eV／316 原子超胞；每个起点取四条轨迹中的最好降幅，再求平均。± 为起点重采样标准误差。
这是已保存结果的复算，不会调用模型或提交作业。

| 方法 | 平均降能 ± 标准误差（eV） | 起点数 | 设置 |
|---|---:|---:|---|
| SFT v2 | 1.714 ± 0.174 | 24 | 旧表示 |
| GRPO 元流程第 3 步 | 1.853 ± 0.167 | 24 | 旧表示 |
| 局部特征贪心 | 2.561 ± 0.170 | 24 | 候选菜单，非 LLM |
| 同菜单随机 | 1.835 ± 0.196 | 24 | 候选菜单，主要随机对照 |
| SFT v4 | 1.804 ± 0.194 | 24 | 候选菜单 |
| GRPO v4 | 1.851 ± 0.168 | 24 | 候选菜单 |

## 配对比较

预先记录的探索门槛为 `max(0.10 eV, 1 × 配对标准误差)`。通过探索门槛不等价于独立确认。
旧随机游走走另一套搜索代码路径；当前主要随机对照是同框架 menurandom。

| 候选相对对照 | 平均差（eV） | 配对 SE | 探索规则 |
|---|---:|---:|---|
| meta_s3_eval − meta_s0b_eval | +0.139 | 0.139 | 达到 |
| meta_s3_eval − meta_s6_menurandom | +0.018 | 0.141 | 未达到 |
| meta_s6_greedy − meta_s6_menurandom | +0.726 | 0.140 | 达到 |
| meta_s6_sft_eval − meta_s6_menurandom | -0.031 | 0.111 | 未达到 |
| meta_s6_sft_eval − meta_s6_greedy | -0.757 | 0.156 | 未达到 |
| meta_s6_eval − meta_s6_menurandom | +0.016 | 0.125 | 未达到 |
| meta_s6_eval − meta_s6_greedy | -0.710 | 0.127 | 未达到 |
| rep_s2_greedy − rep_s2_menurandom | +0.796 | 0.155 | 达到 |
| rep_s3_greedy − rep_s3_menurandom | +1.014 | 0.118 | 达到 |

## 逐步 GRPO 的开发菜单读数

| 配置 | 学习率 | 每轮更新次数 | C |
|---|---:|---:|---:|
| SFT v4 | — | — | 0.0217 |
| 985（原配方） | 2e-5 | 8 | 0.0216 |
| 988（原配方重复） | 2e-5 | 8 | 0.0322 |
| 989 | 2e-5 | 16 | 0.0251 |
| 990 | 5e-5 | 8 | 0.0238 |
| 991 | 5e-5 | 16 | 0.1071 |
| 992 | 1e-4 | 8 | 0.0227 |
| 993 | 1e-4 | 16 | 0.4055 |
| 994 | 3e-4 | 8 | 0.3941 |
| 995 | 3e-4 | 16 | 0.2636 |

原配方重复差为 0.010654，探索门槛为 0.05；最高配置相对原配方提高 0.383958，达到预登记的可区分门槛。
这是开发集选参规则，不是显著性检验。C 是贪心分数优势的捕获比例，不是能量或正确率。
完整定义与解释见 [网格更新](menu-grid-2026-10-28.md)，机器可读复算见 [摘要](../../experiments/diagnostics/menu_grid_summary.json)。

## 当前判断

- 两参数特征贪心在种子 2/3 的已保存评估中也优于同菜单随机，比较仍限于 MACE。
- 菜单 v4 的已测 SFT 与回合级 GRPO 没有达到同时超越随机和贪心的门槛。
- 逐步 GRPO 网格改变了贪心偏好读数；物理搜索收益需以后续完成的物理评估判断。
- 旧表示与菜单随机的比较是事后跨动作接口比较，应与 v4 的同菜单对照区分。
- 当前验证池被反复用于选择方案，后续确认应使用未参与选择的起点和多个种子。

## 全部已保存作业

这一索引包含采集、训练、rollout 和验证等不同阶段；不能把训练中的收益当独立测试表现。

| 作业 | 阶段 | 状态 | 起点数 | 平均降能（eV） |
|---|---|---|---:|---:|
| [diag_randomwalk_val24](../../experiments/results/diag_randomwalk_val24.json) | search | COMPLETED | 24 | 1.627255 |
| [grid_readout](../../experiments/results/grid_readout.json) | — | COMPLETED | 0 | — |
| [imit_greedy](../../experiments/results/imit_greedy.json) | — | COMPLETED | 0 | — |
| [meta_s0_eval](../../experiments/results/meta_s0_eval.json) | blind | COMPLETED | 8 | 1.558126 |
| [meta_s0b_eval](../../experiments/results/meta_s0b_eval.json) | blind | COMPLETED | 24 | 1.714231 |
| [meta_s1_eval](../../experiments/results/meta_s1_eval.json) | blind | COMPLETED | 8 | 1.571968 |
| [meta_s1_train](../../experiments/results/meta_s1_train.json) | round | COMPLETED | 16 | 1.902219 |
| [meta_s2_eval](../../experiments/results/meta_s2_eval.json) | blind | COMPLETED | 8 | 1.408720 |
| [meta_s2_train](../../experiments/results/meta_s2_train.json) | round | COMPLETED | 16 | 2.027186 |
| [meta_s3_eval](../../experiments/results/meta_s3_eval.json) | blind | COMPLETED | 24 | 1.853453 |
| [meta_s3a_train](../../experiments/results/meta_s3a_train.json) | round | COMPLETED | 16 | 1.708828 |
| [meta_s3b_train](../../experiments/results/meta_s3b_train.json) | round | COMPLETED | 16 | 1.683540 |
| [meta_s4_eval](../../experiments/results/meta_s4_eval.json) | blind | COMPLETED | 24 | 1.850678 |
| [meta_s4a_train](../../experiments/results/meta_s4a_train.json) | round | COMPLETED | 16 | 1.820408 |
| [meta_s4b_train](../../experiments/results/meta_s4b_train.json) | round | COMPLETED | 16 | 1.514561 |
| [meta_s5_eval](../../experiments/results/meta_s5_eval.json) | blind | COMPLETED | 24 | 1.651313 |
| [meta_s5_sft](../../experiments/results/meta_s5_sft.json) | sft | COMPLETED | 0 | — |
| [meta_s5_sft_eval](../../experiments/results/meta_s5_sft_eval.json) | blind | COMPLETED | 24 | 1.789514 |
| [meta_s5a_train](../../experiments/results/meta_s5a_train.json) | round | COMPLETED | 16 | 1.790604 |
| [meta_s5b_train](../../experiments/results/meta_s5b_train.json) | round | COMPLETED | 16 | 2.110683 |
| [meta_s6_eval](../../experiments/results/meta_s6_eval.json) | blind | COMPLETED | 24 | 1.851380 |
| [meta_s6_greedy](../../experiments/results/meta_s6_greedy.json) | blind | COMPLETED | 24 | 2.561023 |
| [meta_s6_menurandom](../../experiments/results/meta_s6_menurandom.json) | blind | COMPLETED | 24 | 1.835134 |
| [meta_s6_sft_eval](../../experiments/results/meta_s6_sft_eval.json) | blind | COMPLETED | 24 | 1.804234 |
| [meta_s6a_train](../../experiments/results/meta_s6a_train.json) | round | COMPLETED | 16 | 1.741265 |
| [meta_s6b_train](../../experiments/results/meta_s6b_train.json) | round | COMPLETED | 16 | 1.522287 |
| [rep_s2_greedy](../../experiments/results/rep_s2_greedy.json) | blind | COMPLETED | 24 | 2.741522 |
| [rep_s2_menurandom](../../experiments/results/rep_s2_menurandom.json) | blind | COMPLETED | 24 | 1.945830 |
| [rep_s3_greedy](../../experiments/results/rep_s3_greedy.json) | blind | COMPLETED | 24 | 3.033448 |
| [rep_s3_menurandom](../../experiments/results/rep_s3_menurandom.json) | blind | COMPLETED | 24 | 2.019791 |
| [s1_d0sft](../../experiments/results/s1_d0sft.json) | d0sft | COMPLETED | 52 | 1.221527 |
| [s1_frozen_blind](../../experiments/results/s1_frozen_blind.json) | blind | COMPLETED | 2 | 1.755017 |
| [s1_frozen_r1](../../experiments/results/s1_frozen_r1.json) | round | COMPLETED | 16 | 1.867463 |
| [s1_frozen_r2](../../experiments/results/s1_frozen_r2.json) | round | COMPLETED | 16 | 1.797636 |
| [s1_frozen_r3](../../experiments/results/s1_frozen_r3.json) | round | COMPLETED | 16 | 1.713507 |
| [s1_grpo_step_r1](../../experiments/results/s1_grpo_step_r1.json) | round | COMPLETED | 16 | 1.484174 |
| [s1_grpo_traj_r1](../../experiments/results/s1_grpo_traj_r1.json) | round | COMPLETED | 16 | 1.861804 |
| [s1_itersft_blind](../../experiments/results/s1_itersft_blind.json) | blind | COMPLETED | 2 | 1.777292 |
| [s1_itersft_r1](../../experiments/results/s1_itersft_r1.json) | round | COMPLETED | 16 | 2.190801 |
| [s1_itersft_r2](../../experiments/results/s1_itersft_r2.json) | round | COMPLETED | 16 | 1.968855 |
| [s1_itersft_r3](../../experiments/results/s1_itersft_r3.json) | round | COMPLETED | 16 | 1.639905 |
| [s1_search_blind](../../experiments/results/s1_search_blind.json) | blind | COMPLETED | 2 | 1.807398 |
| [s1_search_r0](../../experiments/results/s1_search_r0.json) | search | COMPLETED | 52 | 1.439461 |
| [s1_search_r1](../../experiments/results/s1_search_r1.json) | search | COMPLETED | 16 | 2.145294 |
| [s1_search_r2](../../experiments/results/s1_search_r2.json) | search | COMPLETED | 16 | 1.932681 |
| [s1_search_r3](../../experiments/results/s1_search_r3.json) | search | COMPLETED | 16 | 2.020733 |
| [s2_d0sft](../../experiments/results/s2_d0sft.json) | d0sft | COMPLETED | 52 | 1.323500 |
| [s2_frozen_r1](../../experiments/results/s2_frozen_r1.json) | round | COMPLETED | 16 | 1.984598 |
| [s2_frozen_r2](../../experiments/results/s2_frozen_r2.json) | round | COMPLETED | 16 | 1.514558 |
| [s2_frozen_r3](../../experiments/results/s2_frozen_r3.json) | round | COMPLETED | 16 | 1.728935 |
| [s2_grpo_step_r1](../../experiments/results/s2_grpo_step_r1.json) | round | COMPLETED | 16 | 1.281749 |
| [s2_grpo_traj_r1](../../experiments/results/s2_grpo_traj_r1.json) | round | COMPLETED | 16 | 1.989363 |
| [s2_grpo_traj_r2](../../experiments/results/s2_grpo_traj_r2.json) | round | COMPLETED | 16 | 1.553581 |
| [s2_itersft_r1](../../experiments/results/s2_itersft_r1.json) | round | COMPLETED | 16 | 1.873214 |
| [s2_itersft_r2](../../experiments/results/s2_itersft_r2.json) | round | COMPLETED | 16 | 1.619935 |
| [s2_search_r0](../../experiments/results/s2_search_r0.json) | search | COMPLETED | 52 | 1.493951 |
| [s2_search_r1](../../experiments/results/s2_search_r1.json) | search | COMPLETED | 16 | 1.931257 |
| [step_readout](../../experiments/results/step_readout.json) | — | COMPLETED | 0 | — |
| [tune_it_1e-4_e2_eval](../../experiments/results/tune_it_1e-4_e2_eval.json) | blind | COMPLETED | 8 | 1.281687 |
| [tune_it_1e-4_e4_eval](../../experiments/results/tune_it_1e-4_e4_eval.json) | blind | COMPLETED | 8 | 1.514357 |
| [tune_it_2e-4_e2_eval](../../experiments/results/tune_it_2e-4_e2_eval.json) | blind | COMPLETED | 8 | 1.688534 |
| [tune_it_2e-4_e4_eval](../../experiments/results/tune_it_2e-4_e4_eval.json) | blind | COMPLETED | 8 | 1.717629 |
| [tune_it_5e-5_e2_eval](../../experiments/results/tune_it_5e-5_e2_eval.json) | blind | COMPLETED | 8 | 1.491657 |
| [tune_it_5e-5_e2_train](../../experiments/results/tune_it_5e-5_e2_train.json) | round | COMPLETED | 16 | 1.656146 |
| [tune_it_5e-5_e4_eval](../../experiments/results/tune_it_5e-5_e4_eval.json) | blind | COMPLETED | 8 | 1.492812 |
| [tune_sa_010](../../experiments/results/tune_sa_010.json) | search | COMPLETED | 8 | 1.692589 |
| [tune_sa_020](../../experiments/results/tune_sa_020.json) | search | COMPLETED | 8 | 1.622322 |
| [tune_sa_040](../../experiments/results/tune_sa_040.json) | search | COMPLETED | 8 | 1.677551 |

来源和变换见 [清单](../../provenance/manifest.json)。历史解释及更正见 [元流程日志](../../history/formal12/META_LOG.md)。
