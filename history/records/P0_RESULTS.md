> 审核06已放行；当前执行、预算和结果请见 [CLAUDE06_RESULTS.md](CLAUDE06_RESULTS.md)。本文件保留先前测试记录。

# YSZ P0 实测报告

更新：2026-10-23T13:30:51.845609+00:00（UTC）。来源等级：调度记录、原始输出和解析结果均为 `code`；建议和后续解释为 `inference`。

**状态：Claude05已放行09_L00_ref与GPU07r，改以相对能误差≤0.02eV/胞、最大逐原子力差≤0.02eV/Å检查；P1/P2保持暂停。**

## 五个获批物理计算点

| 计算点 | 当前作业/状态 | CPU核 | 实际墙钟 s | 核时 | SCF步/是否收敛 | Slurm最大任务RSS | E0 eV/胞 | 最大力 eV/Å |
|---|---|---:|---:|---:|---|---|---:|---:|
| 01_S00_sp16 | 1880378 / TIMEOUT | 16 | 1229 | 5.462222 | 9/否（截断） | 488392K | 不可用（未收敛） | 不可用（未收敛） |
| 02_S00_sp32 | 1880379 / COMPLETED | 32 | 997 | 8.862222 | 18/是 | 341244K | -892.160142200 | 0.32997 |
| 03_L00_gamma | 1882469 / CANCELLED by 10271 | 64 | 1721 | 30.595556 | 7/否（截断） | 417832K | 不可用（未收敛） | 不可用（未收敛） |
| 03b_L00_gamma_lrealauto | 1882471 / CANCELLED by 10271 | 64 | 0 | 0.000000 | 未运行 | 未获得有效采样 | 未计算 | 未计算 |
| 04_L01_gamma | 1882470 / COMPLETED | 64 | 3538 | 62.897778 | 20/是 | 421308K | -2997.390853620 | 0.38068 |

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

- 旧P0所有尝试累计实际：**110.715556 核时**；P1实际 **0.000000 核时**；fast实际 **58.666667 核时**；09参照实际 **84.195556 核时**；含已有冒烟合计：**253.680000 核时**。
- P0实际加未完成点预留封顶：**110.715556 ≤216 核时**。
- 加所有未完成点预留及冒烟：**2157.680000 ≤3000 核时**。
- 原31项清单的2888核时只保留为历史规划；现执行上限以最新分阶段账本为准。失败和诊断按实耗记账，不把取消的原任务和替代任务同时预扣。预留不构成重算或扩批授权。
- P2及后续保持暂停；31_S02/32_S06理想格点起步跨位诊断已准备，尚未执行。P0不能回答跨位是否真实，也不能证明MACE总体排序准确。

机器可读：[BUDGET.json](calibration/BUDGET.json)、[分析结果](calibration/P0_ANALYSIS.json)、[最新调度快照](calibration/P0_LATEST.json)、[回收的原始输出](calibration/P0_outputs)。

## Claude04：两项fast对照 [code / inference]

授权原件：[REVIEW_CLAUDE_04_L00.md](REVIEW_CLAUDE_04_L00.md)。05f_L00、06f_L01各128核、192000 MiB、1小时，不自动重排或重试。只有Γ点，使用vasp_gam；LREAL=Auto、EDIFF=1E-5、AMIX=0.2、BMIX=0.0001，其余物理输入及POSCAR与对应旧作业相同。
两项一起提交；旧04仍运行时，06f只等旧04自然结束（afterany），05f可先运行，使本项目始终最多2个作业并发。旧04即使超时也不按旧设置重跑。P1旧的≤40核时自动放行规则已作废；P1及以后保持暂停。

| 点 | 作业/状态 | CPU | 墙钟 s | 核时 | SCF步/收敛 | 每步平均 s | E0 eV/胞 |
|---|---|---:|---:|---:|---|---:|---:|
| 04_L01_gamma | 1882470/COMPLETED | 64 | 3538 | 62.897778 | 20/是 | 165.0432 | -2997.390853620 |
| fast_L00 | 1882635/COMPLETED | 128 | 813 | 28.906667 | 22/是 | 33.9308 | -2996.759046090 |
| fast_L01 | 1882636/COMPLETED | 128 | 837 | 29.760000 | 21/是 | 36.6643 | -2997.466833840 |
| ref_L00 | 1883149/COMPLETED | 128 | 2368 | 84.195556 | 24/是 | 91.1372 | -2996.681533640 |

所有完整SCF步墙钟：[SCF_TIMINGS.csv](calibration/SCF_TIMINGS.csv)。运行中部分输出仅供进度诊断，不用作能量或力标签。

fast单独授权封顶256核时；本次实际加未完成预留为58.666667核时。旧P0的216封顶与fast的256分开记账，全部校准仍≤3000。用户后续选B，未来CPU只保留94原子的23项待审计算，封顶1904核时；316原子3项后续校准转GPU规划，但尚未放行。新预算见[BUDGET_REPLAN.md](BUDGET_REPLAN.md)。

### 新旧L01精度与成本对照

fast减旧L01：E0差-0.075980220 eV/胞，TOTEN差-0.075980220 eV/胞；最大逐原子力差0.00646117 eV/Å，力分量RMSE0.00180931 eV/Å；墙钟加速4.227倍，核时比0.473。
审核04的绝对能差指标保留为诊断，**审核05已改用相对能门槛，不再以此绝对偏移判失败**。只测一个同构型，无法判断此偏移是否在构型相对能中抵消；没有自动采用fast作为生产协议。

### fast协议的构型相对能

E0(L01)−E0(L00)=-0.707787750 eV/316原子胞。

EDIFF是电子迭代停止阈值，除以原子数不能当作已经验证的总能误差。新旧L01对照测的是整套fast设置的合并影响，不能拆出单个参数的贡献。旧L00没有有效DFT能量，因此仅凭一个L01对照也无法量化构型间相对能偏差。
电荷混合不稳是当前待验证诊断；本轮测试新参数是否解决，不能先保证一定修复。最短距离检查排除了明显原子重叠，但不替代势能面或跨位物理判断。

用户后续已选B，并另外批准1/2GPU L01对照合计≤3卡时；CPU与GPU分别记账。GPU进展见[GPU_PILOT_RESULTS.md](GPU_PILOT_RESULTS.md)。P1/P2保持暂停；不把CPU fast收敛当作精度自动放行。
新输入与提交账：[CLAUDE04_JOB_LIST.json](calibration/CLAUDE04_JOB_LIST.json)、[FAST_SUBMISSIONS.json](calibration/FAST_SUBMISSIONS.json)。

## Claude05最新相对能判据 [code / 用户授权]

新增09_L00_ref采用vasp_std、LREAL=.FALSE.、EDIFF=1E-6，仅修改AMIX/BMIX以稳定电荷混合；128核、1.5小时、独立192核时上限。与旧L01组合计算参考相对能。

门槛：|ΔE_fast(L01−L00)−ΔE_ref(L01−L00)|≤0.02eV/胞，两个构型中最大逐原子力差均≤0.02eV/Å；不是把每原子绝对偏移当作通过证据。

```json
{
  "passed": true,
  "status": "CHECKED",
  "energy_tolerance_eV_cell": 0.02,
  "max_atom_force_tolerance_eV_A": 0.02,
  "n_configurations": 2,
  "approval_reference": "REVIEW_CLAUDE_05_fast_gpu.md",
  "reference_relative_energy_eV": -0.7093199800001457,
  "fast_relative_energy_eV": -0.7077877500000795,
  "absolute_relative_energy_error_eV_cell": 0.0015322300000661926,
  "max_atom_force_difference_eV_A": 0.00674609716353092
}
```

GPU恢复与总体结果见[CLAUDE05_RESULTS.md](CLAUDE05_RESULTS.md)。07r完成SCF才可提交08r，P1/P2始终等Claude放行。
