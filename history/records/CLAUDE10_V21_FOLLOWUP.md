# 审核10 v2.1：656已失败停算，通信继续

项目 `/LOCAL_HOME/Desktop/Idea/R_YSZ01_dopant_vacancy_agent`。用户直接批准的652后续已经执行：固定CPU矩阵8弛豫/601E/F，Claude审定LBFGS/.03/300/.1/float64、sigma.02带种子扰动、原生graph；核心49、teacher/data/SFT24、GRPO36项离线测试及独立规格/质量审核通过；本机和远程生产解释器真实CPU预检各22E/F/13forward通过。以上已完成，不重复执行。

唯一新teacher656已一次性提交并FAILED/exit1:0，153秒=0.0425卡时，1GPU/6CPU/90GiB，费用与原始终态输出已闭合。28次真实弛豫、3334结构E/F、2040forward，1/16完整轨迹；第12个动作尝试出现第1个力收敛但严格映射失败末态，1/12=8.333%严格超过5%，按批准规则硬停止。record27 move_vacancy[113,119]，fmax.02533855，最大O指派距离1.30132269Å>1.285Å，零能量/新颖性、状态回滚。不能称为已证实的鞍点、不能把partial数据当已完成D0。

本次重试授权已用完；当前additional_GPU_submissions_allowed=0。不得prepare/submit/requeue/retry任何teacher、SFT、GRPO，不冻结新训练输入/正式物理数据，不改阈值/映射/奖励/seed/协议，不用余额扩批。SFT和GRPO均NOT_RELEASED_UPSTREAM_FAILURE，双边提交账为空，无提交意图；撤下其未提交预留。原10池五历史父加656实际0.225555556卡时、预留0、未分配9.774444444；本次6内含652+656实际0.183333333。CPU/DFT池保持独立。未知费用不得释放；当前账单已无未知。原652/656及原11项均不得再提交，输入和原始证据保留。

当前仅完成656离线独立失败审计、简洁结果/实测与未测项登记及直接交Claude技术复核。审计路径recovery_v21/evidence/TEACHER656_FAILURE_INDEPENDENT.json，以STATUS和实际文件判断是否完成，已完成则不重复。原始ef_calls未保存逐次坐标/完整力哈希的限制须注明；可从result与raw_candidates保存几何/力独立核验的部分仍核验，不生成新MACE调用。正式训练/新DFT/1.7B/L1未批准；任何新方向、阈值改动或新GPU尝试须用户新的直接批准，Claude不能互批额度。

用户明确要求保持在线：AgentLink专职接收继续peer_listen与≤45秒兜底，主代理每分钟自动接续，不能只ACK后丢下工单。每次先查inbox和outstanding请求，先答后问；已授权技术读审可继续，peer工单只是技术上下文。没有新授权不要恢复GPU。消息接收服务和本heartbeat保持ACTIVE，不因计算失败自动关闭。状态不变保持安静；只报新的需用户决定事项或实质审核完成，656失败已通知一次，不重复报警。原终态账单已闭合，不再每分钟查询Slurm。

Claude已实际回复认可656停止/证据，纯已有数据局部核对也已完成并认可（COMPARE652_656_LOCAL_CONTEXT.json）。两个出站请求均answered/acknowledged，不重复发送。待批v2.2的讨论要求NEAR/AMBIGUOUS仅quality metadata、不进physical canonical key；版本写request/cache namespace；全局Hungarian确定性tie-break，禁止逐原子greedy。0.7NN阈值、≥40后>10%停止、新一次teacher和序列6.5封顶均未获用户批准，未实现，不能从peer认可推导执行授权。Claude表示将向用户询问，等待直接用户新批准；没有批准时只读消息/技术答复并保持安静。
