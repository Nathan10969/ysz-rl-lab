# YSZ CPU 校准清单｜Claude02 已放行 P0，后续阶段待复核

日期：2026-10-23（洛杉矶）。本文件与 [修订方案](PLAN.md)、[逐作业 CSV](calibration/JOB_LIST.csv) 共同审阅。

## 一眼可审的结论

- **[用户授权]** 3,000 CPU 核时校准额度已批准；前置条件是作业清单和预计用量先经 Claude 审核。无需再次申请同一额度。**2026-10-23 Claude02 有条件批准；用户已转达首波5项提交授权。后续阶段仍待P0结果复核。**
- **[code]** 冒烟 1880254 已复查：COMPLETED/exit0、16核23秒、18电子步，`E0≈−99.178732 eV`；Zr2Y2O7，11原子。Slurm分配核时 **0.102222**，相对于3,000授权余额为 **2999.897778** 核时。原始输出：[查询记录](calibration/evidence/smoke_remote_query.txt)。平台如有最小计费单位或系数，以账单为准，此处是Slurm核时记账。
- **[inference/输入已生成]** 主任务固定为 **316 原子 Zr92Y16O208、3×3×3、8 mol%**；94原子胞只做诊断。修订包 **31 项**：24单点、4个MACE末态起步的小胞完整弛豫、2个理想格点起步的小胞完整弛豫、1大胞8步短探针。
- **[inference]** 宽预估 **193.6–1856 核时**，不是测得的置信区间；全部墙钟上限合计 **2888 核时**。加冒烟后留 **111.897778 核时**缓冲，仍在3,000内。
- **[code]** 14个实际输入已完成本机冻结MACE弛豫，均达 fmax<0.03 eV/Å。已保存完整POSCAR、INCAR、KPOINTS、sub.sh、原子顺序映射与哈希；没有把POTCAR原文放入本地包。
- **[code；重点审]** 14起点中10个经MLIP弛豫后改变占位等价类；12个小胞末态仍为12类、2个大胞末态仍为2类。我们分开记初态/末态，DFT验证真实末态几何。材料近邻统计不能用初始标签替代。最大起始格点位移约 2.451 Å；映射本身的最大匹配距离见审计JSON，不把它直接当成真实离子迁移。

等级沿用 PLAN：`code`为实际文件/命令结果；`direct_text`为原文；`official_index`为官方目录；`inference`为设计/估算。

## 1. 执行位置、软件与输入

**[code]** 执行端 `hzw-cpu`，分区 `q_amd_share`；作业目录 `~/online1/Jony/runs/01_calib/`。首波仅上传获批5项；未来阶段输入暂存本地。

- 模板来自 `~/online1/Jony/runs/00_smoke/sub.sh`，使用 `module load amd/intel_compiler/2022u2` 与 `mpirun`。
- VASP：`~/online1/Jony/software/vasp.6.5.1/bin/vasp_std`，SHA256 `f51016d290697d9c63e27feb71a278f7127309f6e23696cd6d68c3f1c43a4159`。
- PAW目录：`~/online1/Jony/software/potpaw_vasp/PBE`。顺序严格 **Zr_sv → Y_sv → O**，对应POSCAR物种 **Zr → Y → O**；批脚本在计算前校验三份势与VASP二进制的SHA。
- PAW_PBE日期/价电子/ENMAX：Zr_sv 04Jan2005 /12/229.898 eV；Y_sv 25May2007 /11/202.626；O 08Apr2002 /6/400。520 eV为最高ENMAX的1.3倍。

**[inference，已写入输入]** 固定常规晶格5.14 Å；小胞10.28 Å，大胞15.42 Å。除31/32直接用初始理想格点外，其他候选用同一冻结MACE权重（SHA见PLAN）在CPU float64预弛豫，fmax0.03、最多300步、BFGS maxstep0.1；不是用DFT结果筛选样本。小胞4随机、4未弛豫MACE低能预筛、4 NN边界；**预筛低能不等于弛豫后仍低能**，表中可直接看到重排。

VASP基准：PBE、520 eV、EDIFF1e-6、ALGO Normal、NELM120、ISMEAR0/SIGMA0.05、LASPH开、LREAL关（03b对照为Auto）、ISPIN1、ISYM0；多k点KPAR2、Γ为KPAR1、NCORE4；94原子NBANDS480（752价电子），316原子NBANDS1536（2528价电子），各核数/精度对照固定能带数。CPU每任务1线程，OMP/MKL1线程，内存按队列1500 MiB/核申请（16核约24GB、32核约48GB）。这些资源是否足够仍要由首波实测；OOM不算物理失败。

完整弛豫：ISIF2、IBRION2、POTIM0.3、NSW200、EDIFFG−0.02。大胞短探针同设置但 **NSW8**；即使退出码0，也不自动计为收敛。Γ只做成本/设置诊断，不未经验证就当正式大胞终评。

NCORE/KPAR/能带并行依据：[VASP NCORE](https://vasp.at/wiki/NCORE)、[KPAR](https://vasp.at/wiki/KPAR)、[NBANDS](https://vasp.at/wiki/NBANDS)。这支持设置设计，不支持未测的耗时估计。

## 2. 分阶段清单及停止规则 [inference]

| 顺序 | 内容 | 项数 | CPU核数 | 本阶段硬封顶核时 |
|---|---|---:|---|---:|
| P0_cost | 小胞16/32核、两个大胞Γ及LREAL Auto对照 | 5 | 16/32 | 216 |
| P1_small | 其余11个小胞基准单点 | 11 | 16 | 176 |
| P2_convergence | 三小胞×两项收敛对照 | 6 | 16 | 192 |
| P2_maincell | 两个大胞2×2×2单点 | 2 | 32 | 512 |
| P3_mainpilot | 一大胞8步短弛豫 | 1 | 32 | 256 |
| P4_small_relax | 四小胞MACE末态起步完整弛豫 | 4 | 16 | 1024 |
| P4_crosssite | S02/S06理想格点直接DFT弛豫 | 2 | 16 | 512 |


**ID用于识别文件，不代表执行顺序**：P3_mainpilot 应在P4_small_relax之前，用大胞短步先判断后续DFT弛豫是否承担得起。

**首波5项合计封顶216核时**：01_S00_sp16、02_S00_sp32、03_L00_gamma、03b_L00_gamma_lrealauto、04_L01_gamma。用户已转达Claude批准，本轮仅提交这5项，最多2个并发。第一项同几何16/32核对照固定NBANDS，不默认占满128核。16/32核实测依据实际核时和墙钟比较；本清单其余小胞默认16核，若需改申请配置，先更新清单交审。

首波检查：SCF收敛、电子数/能带/k点、内存、力、Slurm实耗及VASP每SCF步耗时。若超时/内存不足/输入不一致，暂停扩批，保留日志并修订估算。其后做小胞剩余11项，再收敛对照和大胞2×2×2配对、短弛豫；所有阶段累计实际用量与剩余任务封顶均须留在3,000内。

不启用自动重排队、不自动重试；任何重算也计入授权额度。不在登录节点运行VASP、MACE或训练。POTCAR只在已分配计算任务目录内拼接。不得把该CPU模板直接复制为A100模板，因为GPU的并行/资源规则不同。

## 3. 每个作业的设置与用量

| ID | 阶段 | 构型/类型 | CPU核 | ENCUT | k网格 | LREAL | 墙钟上限 h | 预估核时区间* | 硬封顶核时 |
|---|---|---|---:|---:|---|---|---:|---|---:|
| [01_S00_sp16](calibration/01_calib/01_S00_sp16/sub.sh) | P0_cost | S00 / sp | 16 | 520 | 2×2×2 | .FALSE. | 0.5 | 1.33–8.00 | 8 |
| [02_S00_sp32](calibration/01_calib/02_S00_sp32/sub.sh) | P0_cost | S00 / sp | 32 | 520 | 2×2×2 | .FALSE. | 0.5 | 1.60–16.00 | 16 |
| [03_L00_gamma](calibration/01_calib/03_L00_gamma/sub.sh) | P0_cost | L00 / sp | 32 | 520 | 1×1×1 | .FALSE. | 2 | 8.00–64.00 | 64 |
| [03b_L00_gamma_lrealauto](calibration/01_calib/03b_L00_gamma_lrealauto/sub.sh) | P0_cost | L00 / sp | 32 | 520 | 1×1×1 | Auto | 2 | 8.00–64.00 | 64 |
| [04_L01_gamma](calibration/01_calib/04_L01_gamma/sub.sh) | P0_cost | L01 / sp | 32 | 520 | 1×1×1 | .FALSE. | 2 | 8.00–64.00 | 64 |
| [05_S01_sp16](calibration/01_calib/05_S01_sp16/sub.sh) | P1_small | S01 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [06_S02_sp16](calibration/01_calib/06_S02_sp16/sub.sh) | P1_small | S02 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [07_S03_sp16](calibration/01_calib/07_S03_sp16/sub.sh) | P1_small | S03 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [08_S04_sp16](calibration/01_calib/08_S04_sp16/sub.sh) | P1_small | S04 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [09_S05_sp16](calibration/01_calib/09_S05_sp16/sub.sh) | P1_small | S05 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [10_S06_sp16](calibration/01_calib/10_S06_sp16/sub.sh) | P1_small | S06 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [11_S07_sp16](calibration/01_calib/11_S07_sp16/sub.sh) | P1_small | S07 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [12_S08_sp16](calibration/01_calib/12_S08_sp16/sub.sh) | P1_small | S08 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [13_S09_sp16](calibration/01_calib/13_S09_sp16/sub.sh) | P1_small | S09 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [14_S10_sp16](calibration/01_calib/14_S10_sp16/sub.sh) | P1_small | S10 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [15_S11_sp16](calibration/01_calib/15_S11_sp16/sub.sh) | P1_small | S11 / sp | 16 | 520 | 2×2×2 | .FALSE. | 1 | 1.33–8.00 | 16 |
| [16_S00_encut650](calibration/01_calib/16_S00_encut650/sub.sh) | P2_convergence | S00 / sp | 16 | 650 | 2×2×2 | .FALSE. | 2 | 4.00–24.00 | 32 |
| [17_S00_k333](calibration/01_calib/17_S00_k333/sub.sh) | P2_convergence | S00 / sp | 16 | 520 | 3×3×3 | .FALSE. | 2 | 4.00–24.00 | 32 |
| [18_S04_encut650](calibration/01_calib/18_S04_encut650/sub.sh) | P2_convergence | S04 / sp | 16 | 650 | 2×2×2 | .FALSE. | 2 | 4.00–24.00 | 32 |
| [19_S04_k333](calibration/01_calib/19_S04_k333/sub.sh) | P2_convergence | S04 / sp | 16 | 520 | 3×3×3 | .FALSE. | 2 | 4.00–24.00 | 32 |
| [20_S08_encut650](calibration/01_calib/20_S08_encut650/sub.sh) | P2_convergence | S08 / sp | 16 | 650 | 2×2×2 | .FALSE. | 2 | 4.00–24.00 | 32 |
| [21_S08_k333](calibration/01_calib/21_S08_k333/sub.sh) | P2_convergence | S08 / sp | 16 | 520 | 3×3×3 | .FALSE. | 2 | 4.00–24.00 | 32 |
| [22_L00_k222](calibration/01_calib/22_L00_k222/sub.sh) | P2_maincell | L00 / sp | 32 | 520 | 2×2×2 | .FALSE. | 8 | 32.00–256.00 | 256 |
| [23_L01_k222](calibration/01_calib/23_L01_k222/sub.sh) | P2_maincell | L01 / sp | 32 | 520 | 2×2×2 | .FALSE. | 8 | 32.00–256.00 | 256 |
| [24_S00_relax](calibration/01_calib/24_S00_relax/sub.sh) | P4_small_relax | S00 / relax | 16 | 520 | 2×2×2 | .FALSE. | 16 | 8.00–128.00 | 256 |
| [25_S01_relax](calibration/01_calib/25_S01_relax/sub.sh) | P4_small_relax | S01 / relax | 16 | 520 | 2×2×2 | .FALSE. | 16 | 8.00–128.00 | 256 |
| [26_S04_relax](calibration/01_calib/26_S04_relax/sub.sh) | P4_small_relax | S04 / relax | 16 | 520 | 2×2×2 | .FALSE. | 16 | 8.00–128.00 | 256 |
| [28_S08_relax](calibration/01_calib/28_S08_relax/sub.sh) | P4_small_relax | S08 / relax | 16 | 520 | 2×2×2 | .FALSE. | 16 | 8.00–128.00 | 256 |
| [30_L00_pilot8](calibration/01_calib/30_L00_pilot8/sub.sh) | P3_mainpilot | L00 / pilot_relax | 32 | 520 | 2×2×2 | .FALSE. | 8 | 16.00–128.00 | 256 |
| [31_S02_relax_fromideal](calibration/01_calib/31_S02_relax_fromideal/sub.sh) | P4_crosssite | S02 / relax | 16 | 520 | 2×2×2 | .FALSE. | 16 | 8.00–128.00 | 256 |
| [32_S06_relax_fromideal](calibration/01_calib/32_S06_relax_fromideal/sub.sh) | P4_crosssite | S06 / relax | 16 | 520 | 2×2×2 | .FALSE. | 16 | 8.00–128.00 | 256 |


*估算区间只是排期情景，基于16/32核的假设墙钟，不是由11原子冒烟外推出来的测量。硬封顶来自CPU核数×Slurm墙钟上限，实际按运行时长记账；达到墙钟但未收敛的计算不构成有效标签。

## 4. 实际构型与跨位审计 [code]

| 构型 | 选择依据 | 种子 | 原子数 | 初始Y–空位NN/NNN | 映射后NN/NNN | 占位改变 | MACE弛豫能量 eV/胞 |
|---|---|---:|---:|---|---|---|---:|
| S00 | 预先固定随机 | 20261022 | 94 | 0/3 | 0/3 | 否 | -882.364792 |
| S01 | 预先固定随机 | 20261023 | 94 | 2/3 | 3/3 | 是 | -882.635940 |
| S02 | 预先固定随机 | 20261024 | 94 | 4/2 | 0/7 | 是 | -882.558801 |
| S03 | 预先固定随机 | 20261025 | 94 | 3/2 | 1/5 | 是 | -883.013203 |
| S04 | 100池中未弛豫MACE低能 | 20261048 | 94 | 4/2 | 4/2 | 否 | -881.811842 |
| S05 | 100池中未弛豫MACE低能 | 20261075 | 94 | 4/2 | 3/3 | 是 | -882.419891 |
| S06 | 100池中未弛豫MACE低能 | 20261115 | 94 | 4/2 | 0/6 | 是 | -882.796471 |
| S07 | 100池中未弛豫MACE低能 | 20261118 | 94 | 3/3 | 3/3 | 否 | -882.345773 |
| S08 | NN边界 | 20261029 | 94 | 0/2 | 0/3 | 是 | -882.485218 |
| S09 | NN边界 | 20261030 | 94 | 0/6 | 0/6 | 否 | -882.637914 |
| S10 | NN边界 | 20261121 | 94 | 3/2 | 1/5 | 是 | -882.788236 |
| S11 | NN边界 | 20261096 | 94 | 3/3 | 3/3 | 是 | -882.932198 |
| L00 | 大胞固定随机 | 20261023 | 316 | 6/13 | 6/13 | 是 | -2964.337701 |
| L01 | 大胞固定随机 | 20261024 | 316 | 8/16 | 5/16 | 是 | -2964.250822 |


所有NN/NNN“之前/之后”均基于同一个母晶格壳定义；末态通过周期性最短距离的全局原子–格点指派获取。源格点ID与VASP原子顺序在每项 `atom_mapping.json`，完整末态在 `calibration/geometries/`；审计见 [geometry_audit.json](calibration/geometry_audit.json)。若DFT后映射仍有明显歧义，报告几何和匹配质量，不强行用离散标签解释材料规律。

**评分语义 [inference]**：搜索变量c是初始占位，g(c)是固定协议MLIP弛豫后的能量；代理能量不叫“原初态几何能量”。工具返回原占位、实际几何和末态占位三者。CE拟合同一个g(c)并检查跨盆地映射能否拟合；原先理想格点CE的好精度不能替代这个检查。校准本批的DFT能量是对实际末态的独立计算。

L00/L01的MACE弛豫能差仅约 **0.0869 eV/316原子胞**；是接近的构型，若与数值误差相当，应记“排序无法判定”，不将两点Spearman当总体准确度。两构型首先用于真实尺寸成本、力和k网格敏感性检查。

## 5. 事先登记的物理放行门槛 [inference]

- 三小胞的成对相对能量对650 eV/3×3×3的变化各≤0.02 eV/胞；大胞两构型 Γ→2×2×2 的成对能差变化≤0.05 eV/胞，同时看力的设置敏感性。不能把绝对能量共同平移当相对排序误差。
- 12个不同小胞末态：同几何MACE/DFT排序Spearman≥0.7，MACE前3至少2个仍在DFT前6。
- 逐构型力分量RMSE的中位数目标≤0.20 eV/Å、逐原子误差模长95分位目标≤0.50 eV/Å，并列出最大残余力与离群构型。
- 四个MACE末态起步和两个理想格点起步的完整DFT弛豫检查低能优势是否保留以及是否再次跨位；两个主胞不能证明评分器全面可靠，需要随后独立大胞样本。大胞8步探针用于成本和趋势，不伪造完整弛豫标签。

未过门槛时先诊断设置/映射/领域失配，不继续堆RSI数据。若改用DFT微调MACE或专用势，另登记为新的冻结评分器；终评集不能进训练。**本批是校准/潜在共享训练池，不能作为论文未见终评集。**

## 6. 补做的316原子本机测试 [code]

| 设备 | 精度 | 5次热单点中位数 | 一条弛豫 | 步数 / fmax阈值 |
|---|---|---:|---:|---|
| cpu | float32 | 0.2487 秒 | 27.51 秒 | 87 / 0.03 eV/Å，已收敛 |
| mps | float32 | 0.2141 秒 | 28.86 秒 | 87 / 0.03 eV/Å，已收敛 |
| cpu_float64 | float64 | 0.2888 秒 | 30.29 秒 | 87 / 0.03 eV/Å，已收敛 |


一条几何、5次非缓存热单点；本机PyTorch CPU4线程，MPS显式同步。CPU/MPS float32表项可同精度比较，CPU_float64是输入几何准备的参考；这些数字不等于云端CUDA吞吐，不能按核心数转换卡时。原始：[maincell_timing.json](calibration/maincell_timing.json)。

**icet大胞**：324格点/316原子，13簇函数，1,000次canonical MC，整个检查流程约 **1.663 秒**，成分守恒。使用旧小胞理想格点CE系数只验证尺寸/接口兼容，**不声称316原子CE精度已过关**。[记录](calibration/maincell_icet_smoke.json)。

## 7. 对 Claude 审核 Y-1～Y-5 的回应

| 审核点 | 修订/证据 |
|---|---|
| Y-1 主空间过小 | 主任务改为316原子8%，94原子仅诊断；主空间不等价类下限约1.01×10^29。修正“raw/群阶等于精确类数”及“只筛前几百弛豫等于弛豫基态”的表述 |
| Y-2 资源口径 | CPU VASP已就绪；正式LLM/MLIP在A100 CUDA原版MACE，Mac补丁仅开发；CPU/GPU两个池分开，首批3,000核时已授权 |
| Y-3 覆盖主尺寸 | 两个316原子胞各 Γ/2×2×2单点，另一个8步短弛豫；直接验证大胞，不用小团簇冒充 |
| Y-4 定量门槛 | 预登记ρ≥0.7、前3保留≥2/3、能量收敛与力门槛；大胞两点不作总体统计；追加跨位审计 |
| Y-5 CE和轮次 | icet已实跑3×3×3；D0最多64×8步、后续32×8步，原始E/F及卡时封顶；先测CUDA吞吐再定实际动作数 |

## 8. 审核对象与当前状态

审核对象是**资源配置、物理设置、具体构型、分阶段用量和放行门槛**。审核原件见 [Claude02](REVIEW_CLAUDE_02_calibration.md)。5项P0标为`APPROVED_P0`，其余标为`HELD_FOR_POST_P0_REVIEW`。P0后停止扩批，回报实耗、SCF、内存和LREAL的能量/力/速度差；本轮没有L01的Auto对照，不能用单点总能差判断相对能差。31/32从理想初态起步用于跨位一致性诊断；两个局部优化的有无跳位还受盆地与数值路径影响，须连同末态几何、力及能量判断，不以单一现象当成机制证明。

文件：[PLAN.md](PLAN.md)、[JOB_LIST.csv](calibration/JOB_LIST.csv)、[CONFIGURATIONS.csv](calibration/CONFIGURATIONS.csv)、[BUDGET.json](calibration/BUDGET.json)、[本地输入目录](calibration/01_calib)、[完整配置记录](calibration/configurations.json)。本轮检查记录另存 `calibration/VERIFICATION.json`。


## P0实际执行的启动修订（2026-10-23）[code]

父作业内存24/48GB，但旧mpirun的Hydra子步骤只有1CPU和1500M；导致启动OOM。仅改为相同总量整节点内存后，仍只用1CPU，两个诊断计时中止。现用已验证平台PMI2直接`srun`，16/32任务与CPU一致。科学输入未改，总内存/核数未增加，小胞墙钟缩为20/25分钟以把全部诊断留在216核时之内。

`calibration/01_calib`保留旧审核快照，不能直接重提；有效五项运行输入是`calibration/recovery_inputs/`的01p、02p、03p、03bp、04p。生成器已修复；未来阶段仍等P0复核。实测与完整尝试见 [P0报告](P0_RESULTS.md)。
