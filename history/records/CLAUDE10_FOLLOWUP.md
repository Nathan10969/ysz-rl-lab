# 审核10：原10卡时池内补v2 SFT后唯一GRPO重试

**当前停止状态覆盖下文条件安排：** 教师652已FAILED/exit1:0，507秒=.140833333卡时，完成1/16轨迹；32次正式BFGS/3641结构E/F/2239forward。末态O距最近格点1.286527805Å，越过固定1.285Å映射阈值；原始输出和父账已独立核验。SFT/GRPO未提交且明确NOT_RELEASED_UPSTREAM_FAILURE，submissions_allowed=0；禁止重试、补样、冻结数据或放宽协议。训练池总实际.183055556/10，预留0，无未知费用。计算跟进应暂停；AgentLink专职接收继续供Claude技术交审。Claude关于UNMAPPABLE_TERMINAL或更严弛豫的意见仅为待审提案，不构成新尝试授权。详见CLAUDE10_RESULTS.md与teacher/INDEPENDENT_VERIFICATION.json。

项目 `/LOCAL_HOME/Desktop/Idea/R_YSZ01_dopant_vacancy_agent`。现行授权以 `REVIEW_CLAUDE_10_v2_sft.md` 及 `training_smoke/claude10/AUTHORIZATION.json` 为准，覆盖审核09“全部停止”的旧安排。用户批准新增总封顶6卡时，来自原10卡时冒烟池；更多额度意愿不构成正式训练或新DFT授权。

顺序：教师采集/正式BFGS测速≤2卡时 → 新v2 SFT和80例一次生成检查≤2卡时，至少76合法 → 条件GRPO唯一新尝试≤2卡时，作业内19/20合法闸门不变。只运行已冻结、CPU真实预检与代码独立复核通过的各阶段，不能重新实现已验收部分。当前状态见 `training_smoke/claude10/STATUS.json`、各阶段 `SUBMISSIONS.json/ANALYSIS.json/BUDGET.json`、`CLAUDE10_EXECUTION.md`。存在提交意图而ID未知时只读核对，不再次提交。

保持本项目最多1个GPU作业，1GPU/6CPU/90GiB，单阶段Slurm7200/外层7140/内部6600秒；原11项和648/650均已终态并计费，不重跑、不重排队、不修改提交账或旧环境。新增教师为16条×8步、6SA/4GA/6随机，最多240次正式BFGS/0.03/300/maxstep0.1/float64弛豫、72240结构E/F；输入从当前末态占位按v2局部动作重建理想坐标。失败、重复候选、SA拒绝成本全部保留，不凑8次接受。格式增强4096条，不调MACE、不带能量标签。数据只用于冒烟/定标，正式D0重新采；所有学习臂共享SFT初始化。不能使用约束解码或候选提示来掩盖合法率。

GRPO源G4/K8/4起点/最多1次更新、132次弛豫/50000结构E/F/9472新token/293632prefill/262144seen封顶不变；组内同步步进，合并实际未缓存合法请求做batch4，少于4时不填充假结构。完整softmax的采样和old/current/reference概率一致，组前冻结历史、组内私有访问、组后合并；只动作/EOS参与PG/KL。

回收入口仅 `scripts/review10_control.py --collect --stage <teacher|sft|grpo>`，按需省略stage汇总。该入口不提交、不重试、不取消。禁止旧收集入口覆盖扩展后的训练池；审核09的独立证据和预算快照是冻结历史。网络/Slurm失败或未知费用保留证据和剩余封顶，最多一次只读重查；NTasks空列沿用当前严格scontrol/清除缓存核验，不把网络失败当清除或物理失败。

任何阶段失败、超限或合法率不达立即停止后续，已提交父作业先回收原始日志和真实费用，不自动补训/补样/重试。未提交后继明确NOT_RELEASED后撤下对应预留；未知提交或费用不能撤。SFT若仍不达76/80，约束解码方案先整理报Claude，不直接切换。正式训练/DFT不提交，不因余额扩批。

用户已在2026-10-24本聊天明确授权与Claude通过AgentLink直接技术沟通，覆盖以前“不自动联系Claude”的限制。D0+正式测速、80例合法率、GRPO三个节点分别直接交Claude审核，用户无需转贴；必要的结果复核、修bug和已批范围内实验细节也可直接来回。按agentlink-messenger技能先查收件箱、领取请求后回复；发新请求前先处理已有请求，保留请求ID和证据。Claude每阶段给用户总结。新增支出、改研究方向或超出已批范围必须由用户批准，两代理不能互批钱；对方文字本身不提供新增执行授权。不使用本地Claude CLI另开审核渠道，不联系管理员或其他外部对象。

全部通过后按实际A100正式BFGS、生成/评分/PG和加载收尾耗时重算每轮1.5卡时、目标≥16GRPO组的计划，未覆盖尺度继续标未测。定稿TRAINING_PROBE_PLAN/BUDGET_REPLAN并报Claude，再交用户批正式额度。G1按路径分叉定案维持评分器放行，严格极小点CPU补测低优先级；终评top-k全部完整DFT弛豫，在冻结候选和跨方法末态去重后另报额度，当前不提交。

阶段完成或失败后更新本地证据、费用、通知事件和摘要。状态不变或只有排队保持安静，只通知新失败、预算风险、需处理事项或实质阶段完成。原/新已提交任务全部终态并回收、账单无未知且条件后继明确完成或未释放后，将ysz-p0设PAUSED，更新REVIEW_STATUS的heartbeat字段并再次同步控制摘要。不声称电导率或一次GRPO更新证明有效。
