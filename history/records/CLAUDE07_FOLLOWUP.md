# 审核07后续回收指令

继续本聊天已获用户批准的审核07。项目 `/LOCAL_HOME/Desktop/Idea/R_YSZ01_dopant_vacancy_agent`。审核07覆盖旧审核06的执行规则。先读 REVIEW_CLAUDE_07_gates.md、CLAUDE07_EXECUTION.md、CLAUDE07_RESULTS.md、TRAINING_PROBE_PLAN.md、calibration/REVIEW_STATUS.json、calibration/claude07/STATUS.json，以及下列三个目录的 SUBMISSIONS.json、BUDGET.json、ANALYSIS.json：calibration/claude07、calibration_gpu/claude07、training_smoke。旧审核06账本只作冻结历史，不能回写当前控制状态。

全部11项已经一次性提交：CPU1890508；DFT GPU617/619/621/623/625/627/629/631；冒烟633/635。不得调用任何submit或prepare脚本，不得重复提交、重排队或自动重试，不改变物理参数、时间/卡数上限、QOS或账号。旧L01及24/25/26/28/30保持取消。不得发送Claude、管理员、AgentLink等对外消息，不启动正式RSI、baseline或1.7B训练，也不因剩余额度而追加任务。

三个池严格独立：CPU累计3000核时，历史450.026666667，本轮C1硬限32，累计最坏482.026666667；DFT GPU累计40卡时，历史10.538055556，本轮八项硬限27.5，累计最坏38.038055556；训练冒烟另批10卡时，当前只安排633 SFT≤0.50和635 MACE≤0.75，硬预留1.25，8.75未分配。失败/超时/启动成本均计父作业，不重复计MPI子步骤。GPU账用实际AllocTRES卡数×ElapsedRaw，CPU账用父CPUTimeRAW，卡时不能加进核时。

实际GPU总并发最多2卡。八项DFT为两条afterany资源链，尾629/631覆盖全部八项；633/635均等待afterany:629:631，所以DFT优先。GPU任务各1卡单进程6CPU，DFT为1MPI×6线程；CPU补跑32核。资源链不是科学依赖；普通失败只停止同组真正依赖它的PENDING后继。当前九项DFT及两项冒烟科学上相互独立，普通失败不连带取消。只有明确超封顶、错误分配或异常消耗才停受影响池的清单内PENDING，不碰健康RUNNING或其他项目/池。AssocGrpGRES、Dependency等排队原因不是VASP失败。

按顺序执行：

1. `.venv/bin/python scripts/collect_claude07.py --retrieve --guard`
2. `.venv/bin/python scripts/collect_claude07_smoke.py --retrieve --guard`
3. `.venv/bin/python scripts/render_claude07_report.py`
4. `.venv/bin/python scripts/update_claude07_status.py`
5. `.venv/bin/python scripts/sync_claude07_control.py --apply`

网络/Slurm查询暂时失败或字段缺失时保存证据，最多只读重查一次；不据此声称物理失败、不取消任务、不释放未知费用预留。若新收集器报告DEFERRED/账单不完整/分配核验异常，先核对原始证据再继续相应写控制操作；不能靠重算修复。禁止旧collect_p0.py、collect_gpu_pilot.py、collect_claude06.py、render_claude05_report.py或render_claude06_report.py覆盖现行预算或状态。

终评单点必须COMPLETED/exit0且独立电子收敛，弛豫还须离子力收敛。G1从XML完整帧108续算，续算前免费映射已确认与MACE占位一致；总成本/完整输出数加旧段3.967222卡时/108帧，说明边界重算。G2四项B10/B11/B15/B00比较各自弛豫修正及B10相对能，修正跨度严格<0.10 eV/胞且占位不变才支持单点+等额top-k弛豫抽检，否则全部top-k完整弛豫；报告前三顺序、各项卡时/离子步/占位。G3 B10/B07/B00的k222与原Gamma均为同一MACE几何，最大逐对相对能变化≤0.05 eV/胞才通过。C1 S08/k333不阻塞计划。高精度变体Auto/e5已免费核验；S00偏移修正是三配对共同偏移推断，不能隐藏原False/e6混合设置。

冒烟只测极小Qwen3-0.6B SFT与316原子冻结MACE float64吞吐。633最多64微批/32更新/32768 seen-token，当前数据计划16023 token、最长258，4次greedy生成各≤64新token；验证保存/卸载/重载及有限梯度。635测SP batch1/2/4及等价性，6胞FIRE最多40步/fmax .05，总原始E/F≤1024；step_capped_unconverged保留成本但不是收敛标签。短样本吞吐不能充当2048-token训练实测，有界FIRE不能充当正式BFGS/.03/300步完整弛豫成本。

全部11项终态并回收、三池账核对后，核对原始XML/OUTCAR、逐原子力/末态周期性一对一匹配和冒烟日志，完成CLAUDE07_RESULTS.md、calibration/claude07/SCIENCE.json与BUDGET_REPLAN.md；将可核验冒烟测量填入TRAINING_PROBE_PLAN.md，用对应协议实测重算可完成数据量/卡时/两卡墙钟，未覆盖的长度或正式弛豫参数仍标未测。末态canonical key评分/去重、代理δ≥0.30、主比较等额k=5 DFT低能率不变。正式60/162卡时仅计划，未批准。

向用户回报G1/G2/G3/C1结论、各项实际成本、三个池余额、冒烟实测和失败/未完成/未测项，提交本地成果供Claude人工审核；不自动联系Claude。随后用automation_update把ysz-p0设PAUSED，更新REVIEW_STATUS.json的heartbeat状态并再次同步摘要。科学计算不声称电导率。状态不变或只有排队时保持安静；只通知新失败一次、预算风险、需用户处理的新事项或实质阶段完成。记录已通知事件，历史审核06失败不重复报警。
