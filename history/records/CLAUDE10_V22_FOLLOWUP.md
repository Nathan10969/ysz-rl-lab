# 审核10 v2.2 已停止；定时已删除

SFT661 已 FAILED/exit1:0，24/80 合法，低于 76/80。两分钟自动接续 ysz-v2-2 已实际删除；旧 ysz-p0 也已删除。不得因为历史或排队 heartbeat 再创建、恢复定时或运行 collect。

已闭合：661 实际 1308 秒＝0.363333333 卡时；原总池实际 0.953333333，本轮实际 0.911111111；预留 0，未知费用 0。GRPO 从未提交，未放行且预留已撤；teacher/SFT claims 均已消耗，无自动重试。

仅完成已有数据独立失败审计、结果登记和直接交 Claude 技术复核。AgentLink 专职接收独立继续。任何约束解码、动作表示、概率/闸门更改、失败 adapter 复用或新 GPU 都须用户新的直接批准；peer 意见不等于批准。正式训练、新 DFT、1.7B/L1 仍未批准。

最新状态见 training_smoke/claude10/recovery_v22/{STATUS,AUTHORIZATION,BUDGET}.json 和 RESUME.md。删除前的提示词保存在 evidence/SFT661_DOCUMENTS_BEFORE_FAILURE_REFRESH/CLAUDE10_V22_FOLLOWUP.md，仅供历史查证。
