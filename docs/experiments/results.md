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
| GRPO v4 | 尚无完整本地结果 | — | 候选菜单 |

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

## 当前判断

- 两参数特征贪心在当前验证池上优于旧表示的模型和随机选择器。
- 菜单 SFT 没有达到同时超越随机和贪心的门槛。
- 菜单 GRPO 的状态、是否通过两个对照，以本页表格与保存结果为准；未独立确认前不写成最终结论。
- 当前验证池被反复用于选择方案，后续确认应使用未参与选择的起点和多个种子。

## 全部已保存作业

这一索引包含采集、训练、rollout 和验证等不同阶段；不能把训练中的收益当独立测试表现。

| 作业 | 阶段 | 状态 | 起点数 | 平均降能（eV） |
|---|---|---|---:|---:|
| [diag_randomwalk_val24](../../experiments/results/diag_randomwalk_val24.json) | search | COMPLETED | 24 | 1.627255 |
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
| [meta_s6_greedy](../../experiments/results/meta_s6_greedy.json) | blind | COMPLETED | 24 | 2.561023 |
| [meta_s6_menurandom](../../experiments/results/meta_s6_menurandom.json) | blind | COMPLETED | 24 | 1.835134 |
| [meta_s6_sft_eval](../../experiments/results/meta_s6_sft_eval.json) | blind | COMPLETED | 24 | 1.804234 |
| [meta_s6a_train](../../experiments/results/meta_s6a_train.json) | round | COMPLETED | 16 | 1.741265 |
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
