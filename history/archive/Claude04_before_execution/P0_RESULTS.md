# YSZ P0 实测报告

更新：2026-10-23T11:32:35.652222+00:00（UTC）。来源等级：调度记录、原始输出和解析结果均为 `code`；建议和后续解释为 `inference`。

**状态：L00连续BRMIX错误，已中止；未启动Auto已取消。L01在原授权限额内运行/待回收；P1及以后停止扩批，等待复核。**

## 五个获批物理计算点

| 计算点 | 当前作业/状态 | CPU核 | 实际墙钟 s | 核时 | SCF步/是否收敛 | Slurm最大任务RSS | E0 eV/胞 | 最大力 eV/Å |
|---|---|---:|---:|---:|---|---|---:|---:|
| 01_S00_sp16 | 1880378 / TIMEOUT | 16 | 1229 | 5.462222 | 9/否（截断） | 488392K | 不可用（未收敛） | 不可用（未收敛） |
| 02_S00_sp32 | 1880379 / COMPLETED | 32 | 997 | 8.862222 | 18/是 | 341244K | -892.160142200 | 0.32997 |
| 03_L00_gamma | 1882469 / CANCELLED by 10271 | 64 | 1721 | 30.595556 | 7/否（截断） | 417832K | 不可用（未收敛） | 不可用（未收敛） |
| 03b_L00_gamma_lrealauto | 1882471 / CANCELLED by 10271 | 64 | 0 | 0.000000 | 未运行 | 未获得有效采样 | 未计算 | 未计算 |
| 04_L01_gamma | 1882470 / RUNNING | 64 | 1851 | 32.906667 | 待核验 |  | 待核验 | 待核验 |

SCF指电子自洽迭代；这里只验证单点电子收敛，未做离子弛豫。核时只汇总主作业行，避免把 batch、extern、MPI 步骤重复收费。内存栏是 Slurm 的最大任务 RSS，**不是整个作业所有进程的总峰值**。短任务的0/空值表示未有效采样，不能解释为没用内存。

## 启动排障及额外用量

最初 `mpirun` 经 Intel Hydra 只生成1个CPU的代理步骤。按核申请内存时，它又只继承1500M，而父作业申请的是24000/48000M，导致启动OOM。只改整节点内存解决了OOM，但保留1CPU限制造成异常慢速，因此两个计时中止。最终改用平台已有的 `/opt/hpc/slurm/21.08.6/lib/libpmi2.so` 与 `srun --mpi=pmi2`，步骤核数/任务数实际为16/16和32/32。

修复保持 POSCAR/INCAR/KPOINTS、总申请核数和内存不变；不把错误绑定的运行当性能测量。小胞重跑墙钟分别缩为20/25分钟，所有诊断用量计入原P0的216核时封顶。没有自动重新排队或循环重试。

| 尝试 | 作业 | 状态 | 墙钟 s | 实际核时 |
|---|---|---|---:|---:|
| 01_S00_sp16 | 1880354 | FAILED | 8 | 0.035556 |
| 02_S00_sp32 | 1880355 | FAILED | 11 | 0.097778 |
| 03_L00_gamma | 1880356 | CANCELLED by 10271 | 0 | 0.000000 |
| 03b_L00_gamma_lrealauto | 1880357 | CANCELLED by 10271 | 0 | 0.000000 |
| 04_L01_gamma | 1880358 | CANCELLED by 10271 | 0 | 0.000000 |
| 01r_S00_sp16 | 1880363 | CANCELLED by 10271 | 276 | 1.226667 |
| 02r_S00_sp32 | 1880369 | CANCELLED by 10271 | 173 | 1.537778 |

16核计算偏慢，曾在原30分钟审定范围和216核时以内请求将20分钟改为29分钟；平台拒绝修改运行中限时，**未生效，实际仍为20分钟**。原始记录：[限时请求](calibration/evidence/01p_limit_amendment.json)。

原始证明：[内存诊断](calibration/evidence/memory_diagnosis_stdout.txt)、[CPU诊断](calibration/evidence/cpu_affinity_diagnosis.txt)、[PMI2运行证明](calibration/evidence/pmi2_running_evidence.txt)、[完整尝试清单](calibration/P0_SUBMISSIONS.json)。启动方式依据 [Slurm官方Intel MPI说明](https://slurm.schedmd.com/mpi_guide.html#intel_mpi)。

## 16核与32核，以及 LREAL 对照

- 小胞16/32核：待两个有效结果齐全，目前无法判断。
- 大胞LREAL=.FALSE./Auto：待两个有效结果齐全，目前无法判断。

各设置只有一次运行，节点和共享负载可能不同，速度差是本次观测。L00没有L01的Auto配对，**不能**从L00绝对总能差推出构型间相对能差误差；后续阶段采用哪种LREAL仍需复核。

## 预算与下一道检查

- P0所有尝试累计实际：**80.724444 核时**；P1实际 **0.000000 核时**；含已有冒烟合计：**80.826667 核时**。
- P0实际加未完成点预留封顶：**111.817778 ≤216 核时**。
- 加所有未完成点预留及冒烟：**2783.920000 ≤3000 核时**。
- 31项主清单原墙钟封顶2888核时；失败和诊断按实耗记账，剩余预留按最新重跑限时计算，不把取消的原任务和替代任务同时预扣。剩余预留只是保守预算，不构成重算或扩批授权。
- P2及后续保持暂停；31_S02/32_S06理想格点起步跨位诊断已准备，尚未执行。P0不能回答跨位是否真实，也不能证明MACE总体排序准确。

机器可读：[BUDGET.json](calibration/BUDGET.json)、[分析结果](calibration/P0_ANALYSIS.json)、[最新调度快照](calibration/P0_LATEST.json)、[回收的原始输出](calibration/P0_outputs)。

## Claude03放行规则与补充核实

- 原件：[REVIEW_CLAUDE_03_P0.md](REVIEW_CLAUDE_03_P0.md)。不再复测16核；03/03b/04各64核、1小时、96GB，三个作业合计192核时封顶，最多2并发。
- P1须三个大胞均正常完成且SCF通过，并且03/04各自实际核时≤40；满足后11小胞以32核/30分钟/48GB、每项16核时执行。P2及以后仍待审。
- 旧冒烟1880254的逐步账显示MPI步骤分配了16核，TotalCPU约292秒/23秒墙钟，不能说旧冒烟只跑1核。后续显式资源设置下Hydra子步骤的1核/1500M问题有独立账据。原冒烟模板已备份并改为PMI2，未重跑。证据：[旧冒烟复核](calibration/evidence/Claude03_preflight.txt)、[模板更新](calibration/evidence/smoke_template_update.json)。

### 每个电子步的耗时

完整逐步数据：[SCF_TIMINGS.csv](calibration/SCF_TIMINGS.csv)。下表均为OUTCAR中LOOP的rank0墙钟；实际计费仍按Slurm整作业分配核时。

| 计算点 | 作业 | 电子步数 | 每步平均 s | 每步中位数 s |
|---|---|---:|---:|---:|
| 02_S00_sp32 | 1880379 | 18 | 50.8998 | 50.9318 |
| 03_L00_gamma | 1882469 | 7（未收敛） | 241.3654 | 234.9813 |
| 03b_L00_gamma_lrealauto | 1882471 | 待完成 | 待完成 | 待完成 |
| 04_L01_gamma | 1882470 | 待完成 | 待完成 | 待完成 |

### 同设置下L01相对L00

两个大胞基准结果尚未齐全，暂不计算相对能。

### P1条件判定

当前允许提交：**False**。
{
  "allowed": false,
  "reasons": [
    "03_L00_gamma: awaiting a completed, independently checked SCF result",
    "03b_L00_gamma_lrealauto: awaiting a completed, independently checked SCF result",
    "04_L01_gamma: awaiting a completed, independently checked SCF result"
  ],
  "baseline_actual_core_hours": {
    "03_L00_gamma": 30.595555555555556,
    "04_L01_gamma": 32.906666666666666
  },
  "approval_reference": "REVIEW_CLAUDE_03_P0.md"
}

### 本次停止原因与待复核项 [code / inference]

L00在SCF第5、6步连续报告 `BRMIX: very serious problems`，输出的电子数从设定2528变为2533.72、2565.10；第6步中间总能跳至约−178226 eV。已中止1882469，取消尚未启动的Auto1882471。L011882470未出现同类报错，仍限于已批64核/1小时。没有启动P1，没有重算。
这些是未收敛电子迭代的数值，不能作为材料能量，也不能据此判断结构不稳定。根因尚未确认，需要先复核结构/电子初始化/数值设置，再设计受控修复；本轮不擅自更改设置试跑。
原始诊断：[SCF6诊断](calibration/evidence/Claude03_SCF6_diagnostic.json)、[中止记录](calibration/evidence/Claude03_BRMIX_stop.txt)。自动跟进已按停止规则暂停；L01结果在远端写入，后续只需回收，不能重复提交。
