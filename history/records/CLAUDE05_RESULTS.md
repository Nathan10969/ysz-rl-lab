# Claude05：相对能参照与GPU恢复实测

更新：2026-10-23T13:33:00.409296+00:00。来源等级：原始文件/调度/解析为`code`；解释与预算建议为`inference`。

以[REVIEW_CLAUDE_05_fast_gpu.md](REVIEW_CLAUDE_05_fast_gpu.md)为当前授权。P1/P2及以后暂停；仅09和07r立即放行，08r须07r完成且SCF独立检查通过再提交。不自动重试。
**本批结束：09已收敛，相对能与力判据通过；单GPU已通过CPU一致性，双GPU因MPI槽位启动问题未完成。P1/P2保持暂停，自动跟进已结束/暂停。**


## CPU参照09 [code]

| 作业 | 状态 | CPU核 | 墙钟s | 核时 | SCF步 | SCF通过 | 每步平均s |
|---|---|---:|---:|---:|---:|---|---:|
| 1883149/09_L00_ref | COMPLETED | 128 | 2368 | 84.195556 | 24 | True | 91.1372 |

设置：vasp_std、Γ点、520eV、NBANDS1536、NCORE4、LREAL=.FALSE.、EDIFF=1E-6、AMIX=0.2、BMIX=0.0001。POSCAR和原L00完全相同。128核/192000MiB/1.5小时，单独192核时上限；不是重新放开原失败作业的自动重试。

## fast的相对能判据 [code / 用户授权]

相对能均为同成分、同晶胞下E(L01)−E(L00)。参考取09_L00_ref与旧L01/1882470；fast取05f/1882635与06f/1882636。
门槛：相对能误差≤0.02eV/胞，并且L00、L01两组中最大逐原子力差均≤0.02eV/Å。旧的0.07598022eV/胞绝对偏移只作诊断，不再作为失败判据。

- ΔE_ref=-0.709319980eV/胞。
- ΔE_fast=-0.707787750eV/胞。
- 两者差的绝对值=0.001532230eV/胞。
- 两构型最大逐原子力差=0.006746097eV/Å。
- 当前两构型健全性检查：**通过**。

这只覆盖两个构型，不推广为整个排布空间的精度结论。P1后续三组Auto/.FALSE.配对仍待Claude放行。

## GPU恢复 [code]

| 作业 | GPU | MPI×线程 | 状态 | 墙钟s | 卡时 | SCF步/收敛 | 显存采样峰值MiB/卡 |
|---|---:|---|---|---:|---:|---|---:|
| 554/07r_L01_1g_driverfix | 1 | 1×6 | COMPLETED | 259 | 0.071944444 | 22/True | 10111.0 |
| 556/08r_L01_2g_driverfix | 2 | 2×6 | FAILED | 2 | 0.001111111 | 待完成/False | 0.0 |

07r_L01_1g_driverfix预检：`cuda_preflight_filtered.json`；通过=True，GPU数=1，真实驱动路径：/usr/lib/x86_64-linux-gnu/libcuda.so.570.211.01。

07r_L01_1g_driverfix减CPU06f：总能差0.000000130eV/胞，最大逐原子力差0.000286479eV/Å；门槛分别0.01eV/胞和0.02eV/Å，**通过**；本次墙钟相对CPU06f快3.232倍。

08r_L01_2g_driverfix预检：`cuda_preflight_filtered.json`；通过=True，GPU数=2，真实驱动路径：/usr/lib/x86_64-linux-gnu/libcuda.so.570.211.01。

**08r_L01_2g_driverfix启动失败**：OpenMPI报告`There are not enough slots available ... satisfy the 2 slots`，尚未进入VASP/SCF。平台强制Slurm单任务12CPU，OpenMPI继承1个槽位；显式`--host ...:2`仍未覆盖Slurm槽位限制。CUDA预检识别2GPU且成功，当前是MPI启动问题。
修复草案添加`--oversubscribe`只放宽MPI槽位计数，资源仍是原12CPU/2GPU；在同一个已分配作业内先用2个轻量Python进程验证各绑定6个互不重叠CPU、共12CPU，再允许VASP。草案只保存在本机，未提交；遵守“不自动重试”，等待新放行。

驱动修复先过滤stubs和hpc_sdk CUDA lib64；只有过滤后cuInit失败才用LD_PRELOAD真实驱动。预检失败即退出，不启动VASP。平台强制1个Slurm任务/节点，2卡任务仍只分配12CPU，再在此分配内启动2个MPI进程×6线程；实际MPI/GPU数量以运行日志核验。

## 分池预算 [code]

- CPU含冒烟、所有历史失败与本次09：实际253.680000核时；09剩余封顶0.000000；94原子待审预留1904；累计实耗加全部预留2157.680000≤3000核时。
- GPU含547的4秒失败：实际0.074166667卡时；已提交剩余加未提交08r预留0.000000000；合计0.074166667卡时。审核的名义总封顶2.9511是原精确值2.951111111的四舍五入；脚本3480秒内部截止留足清理时间，实际按更严的2.9511监控，并且始终≤原批3卡时。
- 09是已明确放行的316原子CPU校准例外。其余后续316原子走GPU规划，94原子走CPU；生产成本待有效GPU数据，不自动折算核时/卡时。

当前是否提交P1：**否，待Claude审核**。管理员反馈是附带建议，用户询问后暂放下，未发送任何对外消息。

原始证据：[CPU逐SCF耗时](calibration/SCF_TIMINGS.csv)、[GPU逐SCF耗时](calibration_gpu/SCF_TIMINGS.csv)、[CPU分析](calibration/P0_ANALYSIS.json)、[GPU分析](calibration_gpu/ANALYSIS.json)。
终评独立复核：[四个CPU点的原始XML/EDIFF/输入哈希核对](calibration/evidence/Claude05_final_independent_verification.json)；待审[双GPU修复草案](calibration_gpu/recovery_draft/08s_L01_2g_slotsfix/DRAFT.json)。
