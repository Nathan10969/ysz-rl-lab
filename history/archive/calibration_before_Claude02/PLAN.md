# RSI-MAT-YSZ-01：YSZ 掺杂–空位排布智能体（修订 02）

日期：2026-10-23（America/Los_Angeles）。本版承接 [Claude 审核 01](REVIEW_CLAUDE_01.md) 及用户最新授权；[旧方案归档](archive/PLAN_20261022_before_CPU_smoke.md)。

**当前决定**：主任务为 3×3×3 萤石胞、8 mol% Y2O3，即 **Zr92Y16O208，316 原子、8 氧空位**；原 94 原子胞保留为诊断场。A、C、D′ 仍是备选。

**资源和权限 [code / 用户授权]**：华东 A 的 CPU VASP 6.5.1 已实际通过冒烟测试，DFT 主用 CPU；研究预算规划为 50,000 CPU 核时，当前已授权的校准上限为 **3,000 CPU 核时（含冒烟作业）**。A100 用于正式 LLM/MLIP，约 2,908.78 GPU 卡时是先前余额快照，未实时复核。4090 无账号，不作为执行依赖；VASP-GPU 备用。

**提交条件**：本轮交付本地输入包、作业清单、预计用量和修订方案；必须先由用户转交 Claude 审核，收到审核通过的同步后才可提交。此版本没有提交任何新作业、没有写远端 `01_calib`、没有动其他项目。冒烟 1880254 是已有作业，16×23/3600=**0.102222 核时**，校准余额按 **2,999.897778 核时**记账。

**已直接核实 [code]**：`hzw-cpu` 的模板、INCAR/KPOINTS/POSCAR、OSZICAR/OUTCAR、Slurm COMPLETED/exit 0 和 PAW/二进制 SHA；原始查询在 [smoke_remote_query.txt](calibration/evidence/smoke_remote_query.txt)。CPU 每节点 128 物理核，但本批只申请 **16 或 32 核、单节点**。详见 [校准审阅单](CALIBRATION_REVIEW.md) 和 [逐作业 CSV](calibration/JOB_LIST.csv)。

**目标**：在相同累计 CPU/GPU 成本下，检验多轮自训练的小智能体，是否比搜索、冻结多采样和一次性训练更高效，并在主任务大胞本身接受独立 DFT 核验。不能用局部小胞重算替代大胞总能量验证。

来源等级：`direct_text`＝原论文正文或原始摘要明确写出；`code`＝实际代码、原始文件或本机执行结果；`official_index`＝机构、作者或模型官方目录，未审计全部底层数据；`inference`＝我们的设计、估算或解释，尚非实验结论。下文设计条款统一属于 `inference`，文献和实测另标等级。

**当前建议 [inference]**：继续执行 YSZ 小规模研究，不再铺新方向。先做冻结势的 DFT 校准，再决定是否投入完整 RSI。MACE 的运行速度已过本机可行性检查，物理排序准确性尚未过检查。

## 1. 数据与最近的先例

### 1.1 “2,857 个 DFT 候选”的准确口径

**[direct_text; S1]** Parkes 等，*The atomistic structure of yttria stabilised zirconia at 6.7 mol%: an ab initio study*，PCCP **18**, 31277–31285 (2016)，DOI [10.1039/C6CP04694K](https://doi.org/10.1039/C6CP04694K)。[Imperial 原文 PDF](https://spiral.imperial.ac.uk/bitstreams/1e6c6bd4-69f2-4140-bb45-9435d5b0e9f2/download)；本地 [PDF](sources/Parkes2016.pdf)、[全文文本](sources/Parkes2016.txt)。

- 96 个母晶格格点，4 个 Y 替代 Zr、2 个氧空位，实际 **Zr28Y4O62，94 个原子**。
- 正文报告 8,985 个对称不等价排布，其中仅 **2,857 个**满足 Y 保持八重氧配位，即排除 Y–空位最近邻。这里是受物理先验限制的子空间，不能称为“全部无约束排布”。
- 2,857 个候选都做了固定晶胞的内部坐标 DFT 弛豫。原文更早阶段的 2015 学位论文不能代替 2016 最终计算数据。
- 原文设置：VASP、PBE、PAW，软 O 势、283 eV；33 个随机样本与 400 eV 较硬 O 势比较，相对能量差在约 25 meV/超胞内。固定约 10.23 Å 立方超胞，3×3×3 Monkhorst–Pack k 网格，力小于 0.01 eV/Å，非自旋极化；作者做过测试，所测自旋极化解回到零。
- 正文还写了 4,530,960 种拓扑。**这个中间计数未在本轮复现**，不将其用作新环境的穷举规模。我们自己的带编号占位组合为 C(32,4)×C(64,2)=72,495,360，之后才做对称性去重；这两个数的枚举约定不能擅自等同。[code/inference]

**原始数据可用性 [code/official_index]**：已实际下载原文。Imperial 的 ORIGINAL bundle API 只列一个 2,829,310 字节的论文 PDF，清单保存在 [parkes_original_bundle.json](sources/parkes_original_bundle.json)。未取得 2,857 个结构加逐项能量的数据包；出版社页面访问受限，不能据此断言作者从未公开过附件。**当前不可把这 2,857 条当成已拿到的训练集或离线真值集。** 本轮没有联系作者。

**用途 [inference]**：作为计算设置、NN/NNN 偏好和 `<210>` 空位对结论的文献参照；若以后拿到原始文件，先核实标签、结构和势训练集重叠，再决定作为历史复现集。主终评不依赖取得它。

### 1.2 可用资源盘点

| 资源 | 已核实内容与可得性 | 对本项目的用途 | 来源等级 |
|---|---|---|---|
| Parkes 2016 | 2,857 个受限制 DFT 弛豫候选；原始逐项数据本轮未取得 | 历史物理参照，不充当现成公开 benchmark | direct_text/code，S1 |
| Lee–Prinz–Cai 2010/2011 | 140 个 DFT 构型中 **100 拟合、40 测试**；108 阳离子+216 阴离子格点，共 324 格点；CE+kMC | 经典 CE 基线与输运先例；不照搬其计算耗时 | direct_text，S2 |
| Dalach–Ellis–van de Walle 2010 | DFT+CE+晶格 MC，随浓度分析 Y/空位有序；原始训练构型未取得 | CE+MC 必须比较的直接先例 | direct_text（摘要），S3 |
| Fujii–Kuwabara 2024 GAOP | Table 1 公布 Zr–O/Y–O/O–O Coulomb–Buckingham 参数；拟合纯立方 ZrO2 和 10YSZ；Data availability 写明按请求提供 | CPU 经验势备选、第二评分器；原始 DFT 训练集未取得 | direct_text，S4 |
| Umeno 等 YSZ 偶极势 | NIST 收录、链接 OpenKIM `Dipole_Umeno_YSZ__MO_394669891912_001`；标注适用 2–6 mol% 及氧迁移 | 可获取模型的经验势候选，仍需本地安装与校准 | official_index，S5 |
| Guan 等 2020 Y–Zr–O G-NN/SSW | 原始摘要明确 **28,803 条 first-principles 数据**，从超过 10^7 PES 样本筛选，用于多浓度稳定性与扩散 | 极近的“专用势+大规模搜索”先例；不能声称首次 ML 搜 YSZ | direct_text（摘要），S6 |
| Guan 等 2026 Sc–Y–Zr–O G-NN | 65 种成分、6.7–14.3 mol%，数百万候选；SI 描述模型和数据；本轮未取得可直接加载的权重/逐构型数据 | Sc 转移的强领域对照，优先后续复现其能拿到的部分 | direct_text（摘要）/official_index，S7 |
| PRB 2026 YSZ NNP | 题目为 *Effects of phase transitions and oxygen vacancies on thermal transport…*；原始摘要可读，模型权重/原始数据本轮未证实可下 | 不当成现成可用的缺陷排序器 | direct_text（摘要）/official_index，S8 |
| MACE-MP-0b2 small | 作者 GitHub release 权重已下载，MPTrj 训练，89 元素；本轮在 CPU/MPS 实跑 | **已实测的冻结 MLIP 候选**；只是速度过关 | official_index/code，S9 |
| abICS | 多子晶格 on-lattice NN、主动学习、RXMC/PAMC；论文有 Pt/YSZ 界面例子；代码公开 | 强主动学习/搜索基线；不是 LLM 自训练方法 | direct_text/code，S10 |
| CeO2 Surface Oxygen Vacancy MLFF | ColabFit 目录列 **1,746 构型**，Ce/O 表面，VASP PBE+U+D3、Ueff=5 eV、520 eV、Γ；有原始数据、XYZ 和 HF 链接；本轮未下载核数，直接 XYZ 链接遇 human-check | 是纯 CeO2 表面缺氧集，**不是掺杂 CeO2 体相标签库**，不直接移作我们的任务真值 | official_index，S11 |
| CeO2(111) vacancy CE/ML 工作 | 专门处理多体空位相互作用 | 提醒新基体需要重做电子态和验证协议 | direct_text（摘要），S12 |

**资源检索范围 [code/official_index]**：已检查作者/机构存档、GitHub（MACE、icet、abICS、CatGym）以及 ModelScope。abICS 当前仓库树保存为 [abics_tree.json](sources/abics_tree.json)，commit `b70d5dbc38020d98b8cf05f91cc1045e66a3a750`；按 YSZ/zirconia/ScSZ/CeO2 文件名未找到对应数据包，不能据此断言仓库没有相关内容。LASP 作者站列有 PDF/SI，但实际下载报 SSL/502，见 [失败记录](sources/lasp_downloads.json)。不把“论文报告数据规模”写成“我们已有原始数据”。

**ModelScope [official_index]**：[Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)、[Qwen/Qwen3-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B) 有官方模型入口，HF 原始模型卡也核对过；未下载语言模型。搜到 [OneScience/UMA](https://www.modelscope.cn/models/OneScience/UMA)，索引显示约 193 KB，不能据此认为已包含完整权重，且它不是 facebook 官方发行。本轮不使用这个镜像。未核实到可直接复跑本任务的 ModelScope YSZ 专用智能体。

### 1.3 最近方法与应比较什么

- **[direct_text; S2/S3/S10]** YSZ 的 CE+MC、主动学习构型模型已经存在；我们比较的是训练更新策略的增益，不以“首次自动搜索缺陷”作创新点。
- **[direct_text; S4]** GAOP 的 GA 是优化**势参数**；不能把它写成已完成缺陷排布 GA 搜索。
- **[direct_text; S13]** Banik 等 2021（[arXiv](https://arxiv.org/abs/2106.10557)，DOI [10.1021/acsami.1c07545](https://doi.org/10.1021/acsami.1c07545)）已用延迟奖励 MCTS 搜 MoS2 空位排布。需要考虑多步探索/暂时升能，而不能只保留每步降能的轨迹。
- **[direct_text; S14]** Arrigoni–Madsen 的 CMA-ES+ML 用于稀缺陷连续几何优化，和这里固定占位不同，但属于应承认的进化搜索先例。
- **[code; S15]** [CatGym](https://github.com/ulissigroup/catgym) 提供表面重构 RL 环境和预训练策略；可以借鉴评估和非语言策略，不能拿其合金数字当 YSZ 基线。
- **[direct_text/official_index; S16]** CrystalAgent、Masgent 已有生成/优化工具或缺陷创建工具。当前检索未确认到完整相同的 YSZ 小 LLM 多轮训练实证；这只是检索现状，**不作无人做过的判决**。

## 2. 任务、动作和去重

### 2.1 主任务与诊断场 [inference；构造和计数 code 核验]

| 用途 | 母晶格 | 阳离子/氧格点 | Y/空位 | 实际组成 | Y2O3 mol% |
|---|---|---:|---:|---|---:|
| **主任务** | 3×3×3 常规萤石胞 | 108 / 216 | 16 / 8 | **Zr92Y16O208，316 原子** | **8.000%** |
| 诊断场 | 2×2×2 | 32 / 64 | 4 / 2 | Zr28Y4O62，94 原子 | 6.667% |

电荷补偿：每 2 个 Y 替换 Zr，移去 1 个 O。Y2O3 摩尔分数为 `nY/(2*Ncation-nY)`，不是阳离子中 Y 的比例。固定常规胞 a=5.14 Å，分别为 15.42/10.28 Å 立方超胞；这是跨算法统一的固定晶格模型，尚未标定零压晶格常数。若 DFT 显示系统性大应力，再以校准储备检查胞参数，不能按各方法单独调晶格。

不默认排除 Y–空位 NN，不把 `<210>`、NNN 等答案写成合法性条件。

**构型空间 [code]**：94 原子诊断场的带编号占位为 72,495,360；空间群格点排列数 1,536，故不等价类数至少 47,198。这个除法是下限，不是精确轨道数。按旧 MPS 单点吞吐作线性外推的约 28.6 分钟也只是下限，未计枚举开销，不能称已实测完成穷举。

主任务带编号占位为 `C(108,16)*C(216,8)=524641715979767771108847656044050`，空间群排列数 5,184，不等价类数至少 `101204034718319400291058575626`，约 **1.01×10^29**。审核稿的约 3×10^19 低估了规模，不影响“不可穷举”的选择结论。方法见 [calibration.py](scripts/calibration.py)。

诊断场加入专门的占位枚举+单点排名+同预算弛豫基线；即使穷举了全部固定格点单点，也只保证该评分函数的最小值。仅弛豫单点排名靠前的几百个构型，**不保证已经找到全空间的弛豫基态**。正式主任务必须比较 CE+MC 等非穷举强基线。


### 2.2 动作与观察 [inference]

```json
{"op":"swap_cations","sites":[3,24]}
```

表示交换一个 Y 位和一个 Zr 位；例如实际编号以工具返回的 0–323 主任务格点表（诊断场为 0–95）为准，不固定上述示例对应元素。

```json
{"op":"move_vacancy","sites":[41,77]}
```

表示交换一个空位和一个 O 位。起步允许任意同子晶格两位交换；这是**优化动作，不是物理跳跃轨迹**，不能据此推导迁移时间或能垒。邻近跳跃可以作为统一动作空间的消融。非法 JSON、跨子晶格、同位交换不执行，并记入尝试数、token 和耗时。

每一步可见：晶胞、成分、格点编号/分数坐标、Y 和空位列表、当前与历史最佳的 MLIP 相对能量、最近 4 步动作/能差、已用预算、由同一工具计算的第一至第三配位壳统计。所有方法获得同样的信息；不向智能体泄露 DFT 终评或未来候选能量。若提供候选动作的能量预览，**每个预览也计调用**，不能算免费提示。

初版使用紧凑 JSON 状态而不是整份 CIF。相同状态的坐标轴变换、原点平移和格点重编号做训练增强与推理一致性测试，防止模型只背编号。

### 2.3 对称性与能量函数 [inference；去重已 code 核验]

- 在**无缺陷母晶格**上用 spglib 求空间群操作，并将它们转成格点排列；不是用已经降对称的缺陷结构求操作。
- 对全部等价占位串取字典序最小串作为构型键。诊断场为 **1,536 个**、主任务为 **5,184 个**格点排列。诊断场 100 个随机构型已核对无对称重复；新增主任务的平移不变性和不同占位区分测试。脚本见 [calibration.py](scripts/calibration.py)、[test_calibration.py](scripts/test_calibration.py)；旧小胞探针保存在 [lattice.py](scripts/lattice.py)。
- `X` 只表示 CE/占位表中的空位。送入 MLIP 或 VASP 前移除，保留真实原子的 `site_id`。
- 正式工具将占位先放到标准的母晶格、按构型键固定初始微扰，再做确定设置的 MLIP 内部弛豫；不把上一步残余坐标隐式带到新构型。否则“同一占位”会因不同历史落入不同局部极小值，缓存失效。
- **已发现跨位 [code]**：本批 14 起点中 10 个在 MLIP 弛豫后改变占位等价类（全局最短指派回母晶格），但12个小胞/2个大胞的末态分别仍为12/2个不同等价类，见 [geometry_audit.json](calibration/geometry_audit.json)。初始配位壳不能当末态配位壳；未取得 DFT 前，不把这些位移解释成真实材料迁移。
- **明确评分函数 [inference]**：搜索变量 `c` 是初始格点占位，`g(c)` 是从该标准起点按固定协议做 MLIP 弛豫得到的能量。智能体动作作用于 `c`，工具另返回末态坐标和映回占位；二者不混写为一个状态。CE–MLIP 也拟合同一个 `g(c)`，不能把末态能量误称为原几何的能量；需检查这种含盆地切换的映射是否可被 CE 足够拟合。材料近邻/空位方向分析使用末态；最终 DFT 基于实际末态几何。
- 缓存键包含成分、胞参数、规范占位、模型 SHA、初始几何/微扰、弛豫参数。记录弛豫后是否跨位、是否可映回母晶格、最终几何哈希；离格/跨位构型单独报告并用结构匹配去重，不默默丢掉高分或低分样本。
- 训练/验证按对称性等价类以及轨迹连通的构型家族分组；同时隔离被留作终评的终态，不能只切开相邻两步做随机划分。

## 3. 冻结验证器与独立 VASP 终评

### 3.1 选择 [official_index/code；物理适用性待校准]

**选 MACE-MP-0b2 small**，公开 release、MPTrj 训练、固定 SHA256：

`d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9`

67,622,684 字节，来源和校验见 [manifest](models/manifest.json)。本轮选择它是因为已实际在本机运行，**不是已证明比 UMA 或专用 YSZ 势准确**。MACE 0b3 release 本轮列出的仅 medium；不虚构 small-0b3。UMA 和专用 G-NN 留作未来校准比较，不在这轮再下载所有大模型。

**正式主任务在 A100 使用 CUDA 和原版 MACE，模型权重保持相同 SHA；不把 Mac 的 MPS 补丁部署到云端。云端版本/权重/设备和吞吐要在启动训练前登记，现有环境来自用户同步，尚未在此线程实测。** 本轮为准备输入，在 Mac CPU 用 float64 弛豫；MPS 仅作开发计时。

本机 MACE **0.3.16** 的逐原子输出存在无条件 `.double()`，原版 MPS 报错。本机已在独立 `.venv` 内仅取消该辅助输出的双精度转换；总能量和力表达式未改，权重未改。CPU/MPS 均使用相同修正版本；MPS 载入前在 CPU 将权重转为 float32。修正：[patch](scripts/mace_mps_compat.patch)、[修改前源文件](sources/mace_models_original.py)、[哈希记录](sources/mace_patch_manifest.json)。原始失败保存在 `results/unpatched_mps_console.log`。正式复跑必须带上修正，不能只写“原版 MACE 原生支持”。

### 3.2 数据重叠和独立性的边界 [official_index/inference]

**已经核实**的是 MACE 官方声明的训练数据来源 MPTrj；**没有核实**训练库中全部 YSZ 占位与我们未来终评的重叠。完整训练包约 1.62 GB，本轮只保存 release 清单，未下载审计。不能因为是另一次 VASP 计算，就自动宣布终评从未见过。

正式终评前：冻结势权重和智能体版本 → 固定/封存新浓度与 Sc 任务 → 对候选占位/结构与可取得的势训练库做成分、对称键和结构近邻审计 → 删去精确重复或作为单独的 seen 子集 → 新做封存 DFT。化学体系相同不等于标签泄漏；同一占位的相关弛豫帧或相同参考能量已用于拟合，则不能作为严格未见构型成绩。若训练库审计做不全，报告“训练重叠未排除”，并用可追踪的新构型/专用势作补充。

校准集与共同 DFT 训练池可供所有方法使用；最终 DFT 终评集不参与势选择、SFT、筛轨迹、调参数或停止轮数。D0/R1–R4 的所有轮次可用预先抽定的**诊断集**观察代理偏差，诊断集另计，不混入最终盲评。循环 MLIP 保持冻结；更新评分器是另一个实验。

### 3.3 CPU VASP 校准设置 [inference；本机包及远端模板 code 核验]

| 项目 | 起步设置与校准 |
|---|---|
| 泛函、电子态 | PBE、YSZ 先非自旋极化；先用少量 spin 对照检查。YSZ 不默认加 U |
| PAW | 已读入的 PAW_PBE `Zr_sv 04Jan2005`、`Y_sv 25May2007`、`O 08Apr2002`，ENMAX 分别 229.898/202.626/400 eV；SHA 逐作业校验；不能称为 PAW.54/64 新库 |
| ENCUT | 起步 520 eV，并至少为所用 POTCAR 最大 ENMAX 的约 1.3 倍；3 个代表构型补 650 eV 对照，以相对能量收敛为准 |
| k 点 | 小胞基准 Γ 中心 2×2×2，3 构型补 3×3×3；两个主任务大胞均做 Γ 与 2×2×2 配对。大胞 Γ 先测成本，未通过成对相对能量/力收敛检查前，不当正式终评设置 |
| 电子收敛 | `PREC=Accurate, EDIFF=1E-6, ISMEAR=0, SIGMA=0.05, LASPH=.TRUE., LREAL=.FALSE., ISYM=0`；检查绝缘性、SCF 收敛和设置敏感性 |
| 离子弛豫 | 固定共同晶胞，`ISIF=2, IBRION=2, NSW=200, EDIFFG=-0.02` 起步；关键结构加严到 0.01 eV/Å |
| 单点 | 同一 MLIP 弛豫几何上 `NSW=0`，测代理能量偏差和力；不能单凭单点证明构型已经稳定 |
| 主终评 | **DFT 内部弛豫后能量**为主；同组成/同晶胞内比相对能量。每个浓度独立参考，不能直接比较不同成分的裸总能量 |
| 收敛容差 | 小胞比较三构型的成对相对能量变化，目标 <0.02 eV/胞；大胞 Γ→2×2×2 的两构型相对能量变化目标 <0.05 eV/胞。绝对总能量随截断能的共同漂移不能当相对误差；不达标则加严设置或提高可辨阈值 |
| 并行与能带 | NCORE=4；多 k 点 KPAR=2、Γ 单点 KPAR=1；94 原子 NBANDS=480，316 原子 NBANDS=1536，所有核数对照固定 NBANDS；OMP/MKL 线程=1。首个小胞做 16/32 核配对，按实际核时效率选择，不默认 128 核 |

校准出现明显反向排序、MACE 最优群在 DFT 中整体失去优势，或几何残余力明显异常时，先更换/校准评分器，不继续用它累积自训练数据；6 个弛豫的小样本只能发现严重问题，不能证明整个空间可靠。[inference]

**316 原子大胞的独立 DFT 能量/力及少量直接 DFT 弛豫必须存在**。用局部团簇或更小周期胞重算会改变长程静电、应变与周期性，不能替代大胞排序结论。若大胞成本超出预算，缩小论文主张或调整主任务，不以小胞替代验证后仍声称大胞成功。

主终评采用“单点诊断 + DFT 内部弛豫”，不只做单点。不同算法生成的终态使用完全相同的流程；不同局部极小值/收敛失败计入结果。对最优和几个边界构型做第二初始微扰检查，计入储备。此设计检验固定胞构型优化，不能据此宣称全相图、真实高温平衡或全局基态。

## 4. Baseline 与公平预算 [inference]

**两个横轴一起报告**：(1) 累计原始 MLIP 能量/力评估数；(2) 同一台机器上的累计 CPU/MPS 时间，并分开列训练、推理、搜索与预处理。一次弛豫可能含上百次评估，不能把一次弛豫和一次单点都记成“一次调用”。相同时间上限允许 CE 做更多便宜 MC 步；不能用固定 MC 次数人为削弱它。

| 方法 | 起步实现与公平约束 |
|---|---|
| 诊断场穷举 + 筛选 / 主任务随机重启 | 小胞使用专用占位枚举器，不盲目生成 7,249 万占位后逐个做 1,536 次规范化；记录实际等价类数和成本。主任务不穷举，只随机重启；两者都使用同样的弛豫和合法性规则 |
| 点电荷/Ewald + 筛选 | Y3+/Zr4+/O2−，未弛豫格点上计算，按能量排序后统一 MLIP/DFT 核验；S1 的直接简单对手，不能漏掉 |
| 模拟退火 SA | 同交换动作和冻结 MLIP；预设验证任务上搜索温度表/重启数，把调参成本算进去；允许升能步骤 |
| GA | 固定成分的交叉、修复与交换变异；验证集挑种群/变异率，所有修复、打分、重复候选耗时都记账 |
| **CE–MLIP + MC** | 用同预算 MLIP 标签拟合 icet，多子晶格、空位显式；短程 CE 与加 Ewald/长程项版本都验证；MC 结束候选回到同 MLIP 弛豫并核验 |
| **CE–DFT + MC** | 经典物理对照；共享约 40 个起步的 DFT 内部弛豫标签（确数依校准成本调整），所有学习方法同等可访问；拟合/主动增样/最终精筛成本均计入。不能拿本轮理想几何的 CE smoke 当完整 CE 基线 |
| abICS 型主动学习/搜索 | on-lattice NN/GNN 或 CE + 不确定性/多样性增样 + RXMC/PAMC；相同可用标签、预算和终评。原始包及 aenet/MPI 依赖下一阶段再搭，不假称本轮已复现 |
| 非语言策略 | 以同占位图和能量历史训练小 GNN 策略/动作排序器，或延迟奖励 MCTS；同轨迹/标签可见性和训练时间。检验增益来自迭代策略还是语言模型本身 |
| 冻结小模型多采样 | 同初始模型和 D0；在 RSI 的训练时间里允许继续采样/搜索，多余 token/MLIP 调用照计；不能让冻结模型闲置 |
| 一次性 SFT–独立采样 | 自己按相同累计采集预算生成数据，一次训练；总 seen tokens、训练时间上限与 RSI 相同 |
| 一次性 SFT–相同样本 | 从同一个初始模型，用 RSI 最终累计样本池训练一次；匹配 seen tokens/优化步数。它获得的是事后同数据控制，采集成本按完整 RSI 日志计，不宣称是在线竞争策略 |
| 搜索蒸馏 | SA/GA/CE 产生同预算示范，按相同 seen tokens 训练小模型；分别比较一次蒸馏和分轮刷新搜索数据。检验是不是“好老师+压缩”已经够用 |
| **多轮 RSI** | SFT 后仅由本模型提出操作，冻结物理工具评分，筛选并继续训练 4 轮；与上述各项统一停止规则 |

物理信息分两条可比赛道：**代理赛道**所有方法只用冻结 MACE 的标签；**共同 DFT 训练池赛道**所有方法可访问相同 DFT 训练标签，CE–DFT 和 abICS 在这条赛道比较。两条都用另一个封存 DFT 集终评。不能让 CE 单独免费多拿 DFT，也不能把允许 CE 使用的共同训练池说成终评泄漏。

共同数据生成的物理开销在 50,000 核时实账只付一次；逐方法报告其使用的标签数量、摊销规则和假设独立复跑所需成本。终评候选并集的 DFT 是测量成本，单列，不拿重复使用标签虚增资源。

预处理模型训练/调参计入总时间。预训练公共权重对所有 LLM 方法相同，单独披露，不把已有大模型训练成本记作零成本原创方法。至少 3 个训练/搜索种子用于探针；正式实验优先扩到 5–10 个独立种子。固定同起点、同预算的配对测试；置信区间以独立训练/搜索运行作为重复单位，不能把同一条轨迹的相邻状态当成独立样本；目标材料种类少时只作条件性结论。主要比较 best-so-far 能量曲线、达到预设能量档的成本、DFT 后每个方法等额候选配额的低能率及多样性。

## 5. 主任务训练与 GPU 卡时 [inference，未执行语言模型训练]

正式使用 A100（每卡配 6 CPU 核、90 GiB），CUDA 原版 MACE；Mac 仅开发。基座 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，关闭 thinking，只输出动作 JSON；LoRA rank16/alpha32/dropout0.05，先 q/v 投影，序列上限 **2,048 token**、batch1/累积8/梯度检查点、学习率起步 1e-4。初次 100 步测速后可在验证集调整批大小，所有对照匹配实际 token 与优化步数。1.7B 只做后续规模消融。

输入不展开 324 个原子的长 CIF：给出固定晶格模板、16 个 Y/8 个空位编号、配位壳与局部邻域、能量历史和剩余预算。邻域查询对所有方法相同，任何候选能量预览都计费。模板/坐标变换和编号增强检验是否只是记住索引。

| 阶段 | 原始 MLIP E/F 评估封顶 | 轨迹/动作上限 | 训练 seen-token 上限 | 训练卡时上限 | 采样 MLIP 卡时上限 |
|---|---:|---|---:|---:|---:|
| D0：大胞搜索示范 + SFT | 65,536 | 最多 64 轨迹×8 操作 | 524,288 | 4 | 4 |
| R1：自采样 + 更新 | 32,768 | 最多 32×8 操作 | 524,288 | 4 | 4 |
| R2 | 32,768 | 同 R1 | 524,288 | 4 | 4 |
| R3 | 32,768 | 同 R1 | 524,288 | 4 | 4 |
| R4 | 32,768 | 同 R1，随后冻结 | 524,288 | 4 | 4 |
| 单种子合计 | **196,608** | 动作上限非必达数 | **2,621,440** | **20** | **20** |

以上是可审计的封顶设计，不是 A100 实测耗时。原始 E/F 调用、卡时、动作次数任一上限先到即停；弛豫内部每次评估都计数。先测 A100 的 316 原子单点/弛豫分布，以及语言模型实际吞吐，再调每轮可完成动作数。不能把 Mac 的时间按核数/卡数线性兑换成 A100 时间，也不能假定 512 次动作都能完成。

D0 只用主任务同可见信息的 SA/GA/随机/CE 搜索示范。R1–R4 仅本模型提出操作、冻结物理工具评分，分起点筛高终回报且不同构型的轨迹；允许有价值的升能中间步，保留25% D0、25%早轮、50%新轮回放。seen-token 包含输入上下文，动作 token 单独报告，不混淆唯一数据量和重复训练量。

先 3 个种子探针，正式争取 5–10 个。必须比较：冻结模型继续多采样；一次 SFT 独立采集与相同最终样本池两种；一次和分轮搜索蒸馏；非语言小策略；CE+MC；对称增强与回放消融。高质量搜索老师就能解释的收益不能全部归给 RSI。

这些 GPU 分配是研究预算规划；未授权本轮在 A100 提交训练/评分作业。本轮已在 Mac 对实际 316 原子胞补做开发计时，原始结果在 [maincell_timing.json](calibration/maincell_timing.json)，不称为云端 CUDA 性能。


## 6. 分级迁移 [inference]

| 等级 | 任务 | 评价边界 |
|---|---|---|
| 诊断 | 94 原子、6.667%，工具/穷举/代理核验 | 不作为主任务成功证据 |
| L0 主任务 | 316 原子、8.000%，新起点、编号/坐标变换 | 同一景观的搜索和不变性；不等于新材料迁移 |
| L1 | 同 3×3×3，20Y/10空位＝Zr88Y20O206，314 原子，10.204 mol% | 预先封存，不训练/调参；应有大胞本身 DFT 终评，每种成分独立相对能量参考 |
| L2 | 同主胞换 Sc 掺杂 | 评分器和 Sc PAW 另校准，之后才给有限 DFT 配额；本批仅 Zr/Y/O |
| L3 | 新基体如掺杂 CeO2 | 第一阶段排除，Ce 4f/U/自旋与局域态需重新设计 |

若只在 L0 成立，主张限于主任务景观中的有效搜索；若没有 L1 的独立结果，不称跨浓度泛化。大胞终评太贵时，不把小胞或局部截取验证解释成大胞迁移成功。


## 7. 材料问题与可支持的解释

**[direct_text; S1/S2/S3/S7]** 文献对 Y–空位 NNN、空位对方向、有序聚集及电导的讨论高度依赖浓度、约束、胞形与温度。S1 的 `<210>` 是固定小胞、6.7% 和受限候选中的低能结论，不应当作全部 YSZ 的普适常数。

**计划统计 [inference]**：按母晶格配位壳，比较 Y–空位 NN/NNN/第三壳、Y–Y、空位–空位的数量/条件概率；对相同组成随机排布作富集比参考。空位对用周期性最小像，并记录等距离多重像，按对称方向族归类，避免任意选一个等长向量。分别画 MLIP 入选集与 DFT 后低能集，检查偏好随训练轮数是否只在代理上增强。报告不同种子和独立构型家族的不确定度；从搜索样本频率不能直接推导热平衡概率。

**若声称提高离子电导，至少另需 [inference]**：代表性 O 跳跃的 DFT NEB/势能垒核验；足够连通的跃迁网络与 kMC，或校准过的长时间多温度 MD；扩散统计、相关运动/电导换算、有限尺寸和采样收敛。低构型能量、某条短 NEB 或“空位更分散”都不等于电导更高。本轮主论文先不承诺电导提升。

## 8. CPU/GPU 双预算与物理放行门槛 [inference]

### 8.1 CPU DFT：总研究规划与当前授权

| 部分 | 研究规划核时 | 状态 |
|---|---:|---|
| 校准 | 5,000 | **仅首批累计 3,000 已授权**，还须 Claude 审清单；另 2,000 是规划储备 |
| 探针/共同 DFT 池 | 7,500 | 校准后再定构型规模和数量，不能默认完成 40 个大胞标签 |
| 主实验/独立大胞终评 | 27,500 | 先争取 12–24 个去重大胞终态，实际数量按 DFT 弛豫成本；配额不足时收缩主张 |
| 储备 | 10,000 | 失败重算、加严收敛、必要的 CE 增样/迁移校准，未触发不花 |
| 合计 | **50,000** | 研究规模规划不是无限制提交授权 |

首批新清单共 **30 项**，宽估算 **185.6–1,792 核时**（尚未大胞实测），逐作业墙钟全部跑满合计 **2,824 核时**；加已有冒烟后仍留 **175.897778 核时**缓冲。首波4项硬封顶 **152 核时**。这些是申请/墙钟封顶，不是会预扣或必定花掉的量。

首批见 [CALIBRATION_REVIEW.md](CALIBRATION_REVIEW.md)：12 个小胞基准单点，1 个重复的 32 核计时，6 个小胞收敛单点，6 个小胞内部弛豫，两个 316 原子胞的 Γ/2×2×2 单点配对，以及一个大胞 **8 步**短离子步成本探针。完整输入包保存在 `calibration/01_calib/`，不含赝势原文，不含自动批量提交脚本。

已有冒烟作业消耗 **0.102222** 核时；新清单逐作业墙钟上限换算出的总封顶在 `calibration/BUDGET.json`。首波只做 94 原子 16/32 核配对及两个大胞 Γ 计时，再依据真实的 SCF/离子步耗时判断余下阶段。未通过成本/设置检查时暂停扩批，不能让全部任务盲目跑满。

11 原子 Zr2Y2O7 冒烟使用 520 eV、4×4×4 k 点、对称性默认开启，NKPTS=21、NBANDS=52；正式小胞关闭对称性并锁定 NBANDS=480，主胞 NBANDS=1536。**23 秒不能按原子数直接放大，亦不能用其证明 316 原子大胞每个单点只要几分钟**。旧的 8–32 核时小胞单点只是保守情景，新的 CSV 同时列宽估算和硬墙钟上限，待首波实际值替换。

核时记账用 Slurm `AllocCPUS×ElapsedRaw/3600`；排队时间另记。固定胞内部弛豫的最终一次能量/力只在 SCF 与离子停止条件都通过时作为标签；COMPLETED/exit0 本身不够。大胞8步短探针只用于每步成本、力变化和收敛趋势，未达到力阈值不能当已弛豫标签。

### 8.2 GPU 池：只作规划，不提交

按先前约 2,908.78 GPU 卡时快照暂分：LLM 训练/相关推理 **600**；MLIP、搜索与 CE/非语言对照 **600**；备用 VASP-GPU **300**；未分配储备 **1,408.78**。合计 2,908.78。没有 4090 账号也可在 A100 顺序执行。GPU 和 CPU 核时不互换；实际余额、附加 CPU/软件费用、平台计费规则在提交前核实。600 卡时不是某模型已测需求或本轮支出授权。

### 8.3 校准的定量门槛（在看 DFT 前登记）

1. **数值设置**：小胞三代表构型的成对相对能量对 520→650 eV、2×2×2→3×3×3 的变化各不超过 **0.02 eV/胞**；大胞 Γ→2×2×2 的两构型相对能差变化不超过 **0.05 eV/胞**，同时检查力的设置敏感性。未达标则加严，不用绝对总能量的共同漂移作误差。
2. **小胞排序**：12 个实际且相互不同的 MLIP 弛豫末态几何上的 MLIP vs DFT 单点 Spearman **≥0.7**；MLIP 选出前3中至少 **2 个仍在 DFT 前6**。能量采用同组成同设置并减去共同参考；近似简并样本与数值误差另报。
3. **力**：12 小胞的逐构型力分量 RMSE 的中位数目标 **≤0.20 eV/Å**，且逐原子力误差模长的95分位目标 **≤0.50 eV/Å**。同时报告每构型最大 DFT 残余力和离群项，不能仅靠总体平均掩盖失败。
4. **主胞必要检查**：两构型的 DFT/MACE 相对排序符号在可辨误差之外不应反向，且力误差不过上述严重偏差门槛。若能差小于数值容差则记“无法判排序”；**两构型不能产生有意义的总体 Spearman 证据**，大胞评分器全面放行还需后续独立大胞池。
5. **弛豫核验**：六个小胞直接 DFT 弛豫检查原代理低能优势是否保留、是否跨位或落入同一盆地；大胞短探针提供成本趋势，不替代正式大胞弛豫。

这些是“允许开展低成本训练探针”的条件性门槛，不是势已普适准确的证明。12 个样本有预先分层，包含较宽能量范围，相关系数不代表低能尾部的整体质量；后续应补低能区域的独立样本。

若门槛未过：先定位设置敏感性/几何跨位/领域失配；可选可追踪专用势，或共享 DFT 池微调 MACE，另登记为**另一个冻结评分器实验**。不自动用 DFT 盲评数据更新评分器，也不继续堆 RSI 轮数。

### 8.4 研究结果：支持 / 不支持 / 无法判断

同组成预登记能差阈值 `δ=max(0.05 eV/胞,3×数值相对能量误差)`，大胞若误差更大则提高 δ。

- **支持**：在相同累计时间下，RSI 相对最强搜索/CE与同数据一次 SFT，达到相同低能档成本下降约20%，或等额终态配额的独立大胞 DFT 弛豫能量改善超过 δ；独立种子方向一致且不确定度支持，并在 L1 保留收益。
- **不支持**：有足够且数值一致的大胞终评，CE/搜索或同数据 SFT 与 RSI 实用等价或更好；或代理分数提升不转化为 DFT 收益。收缩或停止这条算法主张，不凭更多轮数强行上分。
- **无法判断**：大胞/低能样本不足、区间跨实用阈值、训练重叠未排除、数值误差超过提升、CE 标签不足、弛豫未收敛或跨位严重。先解决证据缺口，不拿小胞验证替代大胞结果。

每个方法按种子/任务预先等额选择终态，取并集统一 DFT；相同构型复用标签但不虚增样本数。置信区间以独立运行/构型家族为单位，不把相邻轨迹状态当独立样本。只声称找到更好的已知结果，不声称大胞全局基态。


## 9. 本机实测：结果、文件与复跑 [code]

环境：M3 Max、64 GB；独立 [requirements-lock.txt](requirements-lock.txt)：Python 3.11.14、torch 2.14.1、mace-torch 0.3.16、ASE 3.29.0、icet 4.0、spglib 2.8.0。CPU 限 4 个 PyTorch 线程，CPU/MPS 顺序运行，MPS 显式同步计时，无 CPU fallback 开关。

结构：Zr28Y4O62、a=5.14 Å、94 原子；同样 100 个随机种子 `20261022..20261121`，无 NN 排除；全部 100 个对称不等价。能量分布为**理想格点单点，未弛豫**，只对第一个排布做一条 CPU 和一条 MPS 弛豫，不伪称做了 100 次弛豫。

| 项目 | CPU | MPS |
|---|---:|---:|
| 5 次非缓存热单点中位数 | 0.05251 s | 0.03632 s |
| 100 个单点总计 | 5.447 s | 3.644 s |
| 固定胞 BFGS 弛豫 | 6.407 s，106 步 | 5.287 s，106 步 |
| 收敛阈值/上限 | fmax<0.05 eV/Å，最多 200 步；已收敛 | 相同；已收敛 |
| 弛豫末能量 | -882.348511 eV/超胞 | -882.348633 eV/超胞 |

CPU 随机 100 单点范围 **-875.998474 至 -872.392944 eV/超胞**，均值 -873.972981、标准差 0.720049 eV；只是该冻结势在这批未弛豫排布的统计。

跨设备最大单点能差 **0.00024414 eV/超胞**，能量 Spearman≈1；弛豫末能差 0.00012207 eV，最大力分量差约 0.00021822 eV/Å、最大位置差约 0.00001854 Å。辅助输出修正前后 CPU 随机能量均值差为 0。见 [cross_device_check.json](results/cross_device_check.json)。该检查不代替 float64 数值敏感性和 DFT 校准。

**最小 CE 实跑**：双子晶格 `[Zr,Y]` 和 `[O,X]`，pair/triplet cutoff 5.2/3.0 Å，13 个簇函数，固定成分下训练矩阵秩 11；ridge 拟合 80 条 MACE 理想格点单点、20 条留出。留出 RMSE **0.07886 eV/超胞**，1,000 次 canonical MC 用时约 **0.0080 s**，成分守恒，末态还用原 MACE 重算。见 [probe_ce.json](results/probe_ce.json)。这是 CE/MC 工具链测试，不能与 DFT 弛豫标签拟合的 CE 精度或基态能力等同。

![随机单点分布与跨设备差异](results/probe_distribution.png)

复跑（工作目录为本项目；新环境先 `uv venv .venv --python 3.11`，再 `uv pip install --python .venv/bin/python -r requirements-lock.txt`；已建环境跳过安装）：

```bash
.venv/bin/python scripts/apply_mps_patch.py
.venv/bin/python scripts/test_lattice.py
.venv/bin/python scripts/probe_mace.py --device cpu
.venv/bin/python scripts/probe_mace.py --device mps
.venv/bin/python scripts/probe_ce.py
```

- 计时/设备：[probe_cpu.json](results/probe_cpu.json)、[probe_mps.json](results/probe_mps.json)、[probe_metadata.json](results/probe_metadata.json)。
- 原始结构/能量：`results/random_100_{lattice,physical}.extxyz`、`results/random_energies_{cpu,mps}.csv`；`results/relax_{cpu,mps}.traj` 是逐步轨迹。
- 代码：[lattice.py](scripts/lattice.py)、[probe_mace.py](scripts/probe_mace.py)、[probe_ce.py](scripts/probe_ce.py)。空位表示/对称去重/合法换位的 3 项单元检查已通过。
- 不在原有吸附项目环境里升级包；新环境和模型权重均在本目录。

## 10. 本轮交付与下一步

1. 用户把 [CALIBRATION_REVIEW.md](CALIBRATION_REVIEW.md)、[逐作业 CSV](calibration/JOB_LIST.csv) 和本方案交 Claude 审。输入/资源/哈希均可逐项审阅，包中没有 POTCAR 原文。
2. **当前 3,000 核时额度已授权，无需重复申请；前置条件是 Claude 审核清单与预计用量通过。** 本轮不提交、不批量上传；远端 `~/online1/Jony/runs/01_calib/` 只作为已指定的未来目录。
3. 审核同步回来后先执行 P0 计时，再按真实核时、SCF/力和设置检查决定后续；超出现清单、上限或计费口径的变更先更新审阅材料。
4. 通过物理校准后再做 A100 CUDA 原版 MACE 与 Qwen 小规模测速和 D0/R1/R2；云端训练与新评分器微调不在本轮已执行范围内。

本轮新增的 316 原子本机开发计时、icet 大胞操作与输入包验证，见 [CALIBRATION_REVIEW.md](CALIBRATION_REVIEW.md) 的实测与核验部分。原 94 原子探针保留在 §9，不能当主任务大胞的已完成研究结果。


## 参考来源与查阅深度

- **S1 [direct_text/code]** Parkes 等，[DOI](https://doi.org/10.1039/C6CP04694K)；已读/下载全文，方法约 manuscript pp.9–11，点电荷 pp.21–22，候选限制 pp.8–10；公开数据包未取得。
- **S2 [direct_text]** Lee/Prinz/Cai，[arXiv:1011.1838v2](https://arxiv.org/abs/1011.1838v2)，[PRB DOI](https://doi.org/10.1103/PhysRevB.83.052301)；已读全文 [本地](sources/CE2010.pdf)。CE 参数化与 kMC 输运需区分；结构搜索用 MC/退火。
- **S3 [direct_text，摘要]** Dalach 等，[PhysRevB.82.144117](https://doi.org/10.1103/PhysRevB.82.144117)。
- **S4 [direct_text]** Fujii/Kuwabara，[10.1016/j.commatsci.2023.112722](https://doi.org/10.1016/j.commatsci.2023.112722)，[大学全文](https://ir.library.osaka-u.ac.jp/repo/ouka/all/93527/ComputMaterSci_233_112722.pdf)，已读全文，参数 Table 1，数据声明 p.6。
- **S5 [official_index]** [NIST YSZ 偶极势目录](https://www.ctcms.nist.gov/potentials/entry/2013--Umeno-Y-Iskandarov-A-M-Kubo-A-Albina-J-M--O-Y-Zr/)。
- **S6 [direct_text，摘要]** Guan 等，[JCP DOI 10.1063/1.5142591](https://doi.org/10.1063/1.5142591)，[原摘要](https://pubmed.ncbi.nlm.nih.gov/33480711/)；作者全文/SI 下载失败已记录。
- **S7 [direct_text，摘要/official_index]** Guan 等，[10.1021/acs.jpcc.5c08205](https://doi.org/10.1021/acs.jpcc.5c08205)；已核对出版者摘要/SI 描述及[作者目录](https://zpliu.fudan.edu.cn/publication/list.htm)，未审计底层数据/权重。
- **S8 [direct_text，摘要]** Zang 等，[PRB 113,174117 (2026)](https://journals.aps.org/prb/abstract/10.1103/ntwm-4gf7)；未取得权重和原始 DFT 集。
- **S9 [official_index/code]** [MACE 官方基础势目录](https://github.com/ACEsuit/mace-foundations)、[0b2 release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mp_0b2)、[MPTrj](https://figshare.com/articles/dataset/23713842)；已下载小势并运行，未审计完整训练包。
- **S10 [direct_text/code]** [abICS 论文全文](https://arxiv.org/html/2309.04769v1)，[DOI 10.1080/27660400.2023.2284128](https://doi.org/10.1080/27660400.2023.2284128)，[GitHub](https://github.com/issp-center-dev/abICS)；§6.3 为 Pt/YSZ 界面；未声称复现其生产计算。
- **S11 [official_index]** [ColabFit CeO2 数据目录](https://materials.colabfit.org/id/DS_puukty1cgl4q_0)，[原始数据 DOI](https://doi.org/10.5281/zenodo.20033455)；目录核对，未下载计数。
- **S12 [direct_text，摘要]** [CeO2(111) vacancy ML/CE](https://arxiv.org/abs/2406.15956)；不同电子态/任务，不作为 YSZ 直接基线数字。
- **S13 [direct_text，摘要]** [Delayed-reward MCTS](https://arxiv.org/abs/2106.10557)。
- **S14 [direct_text，摘要]** [Evolutionary computing and ML for defects](https://arxiv.org/abs/2011.06900)。
- **S15 [code/official_index]** [CatGym 作者代码](https://github.com/ulissigroup/catgym)。
- **S16 [direct_text/official_index]** [CrystalAgent 原稿](https://openreview.net/pdf?id=SOAfumn8bA)、[Masgent 原文](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00043f)；作为相关工具型智能体，不借其结果推断 YSZ RSI 成效。
- **S17 [official_index]** VASP 官方 [ENCUT](https://vasp.at/wiki/ENCUT)、[PAW 目录](https://vasp.at/wiki/Available_PAW_potentials)、[NCORE](https://vasp.at/wiki/NCORE)、[KPAR](https://vasp.at/wiki/KPAR)、[NBANDS](https://vasp.at/wiki/NBANDS)；只支持参数定义与需收敛测试，不支持我们尚未测得的核时估算。
- **S18 [official_index]** [icet 官方文档](https://icet.materialsmodeling.org/)、[Qwen3-0.6B 官方卡](https://huggingface.co/Qwen/Qwen3-0.6B)；icet 已实际跑通，语言模型未训练。
