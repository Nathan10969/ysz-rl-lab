# 审核08/09执行登记

现行依据：REVIEW_CLAUDE_08_RL_arm.md §8、REVIEW_CLAUDE_09_final_protocol.md及用户直接补充。此前登记归档于archive/review08_before_review09/，原11项账本保持冻结。

| 项目 | 结论/数字 | 异常 |
|---|---|---|
| MACE635替代648 | 36s/0.0100卡时，1023 E/F，batch4/16等价通过 | 21个FIRE40副本全部未收敛，不是正式BFGS测速 |
| 唯一GRPO650 | 72s/0.0200卡时，合法率0/20，MACE/PG/更新0 | NOT_RELEASED_LEGALITY_GATE；停止，不再重试或补SFT |
| CPU实际预检 | MACE batch1/4共23 E/F；GRPO batch1/4 SP及5个BFGS单例回调共15 E/F | 共同环境建立时一次提前导入未启动模型；传输中断后主机内复制既有权重并核验34项SHA |
| G1 CPU诊断 | 116 E/F、2弛豫、50.49s；正式理想重启仍原MACE盆地，DFT占位盆地经MACE调整低0.353659eV | 正式协议复现路径分叉；更严极小点未测 |
| G2 | 修正跨度0.14229335，前三B10→B11→B15未变 | 全部top-k完整DFT弛豫；冻结去重后再报额度，当前不提交 |
| 冒烟池 | 633+635+648+650=0.042222222/10卡时 | 失败635费用保留；无在途/条件预留，余额不触发追加 |

正式目标16组/轮、1.5h、最多528次弛豫；D0扩大四倍至64轨迹/576次。正式BFGS/PG A100吞吐未测，不承诺目标完成数，正式包和新增DFT均未批。详细结果及独立证据入口见CLAUDE09_RESULTS.md、各新任务INDEPENDENT_VERIFICATION.json与TRAINING_PROBE_PLAN.md。
