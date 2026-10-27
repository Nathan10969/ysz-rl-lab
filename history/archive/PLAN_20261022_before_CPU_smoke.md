# RSI-MAT-YSZ-01：YSZ 掺杂–空位排布智能体

## 资源口径更新（2026-10-22，以本段为准）

**只读实查补充**：A100 每计算节点 48 CPU 核/8 GPU，严格每 GPU 配 6 CPU 核、90 GiB；CPU 超算每计算节点 128 物理核。账号与即时状态见 [资源核查](CLUSTER_RESOURCE_CHECK_20261022.md)。已连通这两个别名核查资源；尚未远程核实全部软件、赝势和计费规则，也未提交作业。


根据用户最新转交的 Claude 同步，主预算改为约 **2,908.78 GPU 卡时**；正式 DFT 拟用云端 A100 的 VASP GPU 版，小模型训练/MLIP 拟用云端 4090，Mac 用于开发和小型探针。CPU 超算作为备用池，额度待确认。软件路径和余额主要来自同步材料，尚未实测软件可用性和核对实际账单；硬件和调度资源已经通过 SSH 只读核查，见本段核查链接。

**下文 §8 的 50,000 CPU 核时分配和 §10 的首批 3,000 核时申请保留为历史草案，暂停执行，不能当成 GPU 卡时配额。** 3,000 核时从来不是本机已发生的耗时。新的卡时预算需要结合实际申请卡数、平台计费规则和云端 VASP/训练小样本计时重新制定。此次 Mac 仅跑过 MACE/CE，未运行 VASP DFT；本轮仍不提交云端作业。

单位：CPU 核时＝CPU 核数×小时；GPU 卡时＝GPU 张数×小时。两者不能直接一比一兑换。Mac 实查为 M3 Max，16 个 CPU 核（12 性能核、4 能效核）；不能把它们视为与集群 CPU 等速的 16 个核。


日期：2026-10-22（洛杉矶）；本机实测：2026-10-23 UTC。状态：**设计和本机探针已完成；未训练语言模型，未提交 DFT。**

**目标**：用一个小智能体提出原子换位，再用物理计算打分；检验多轮自训练能否在相同累计成本下，比搜索、冻结模型多采样和一次性训练更有效，并在新浓度上保留经 DFT 核验的收益。

**用户决定**：YSZ 为唯一深挖方向；VASP 总预算 **50,000 核时**。A、C、D′ 保留备选，②、③ 暂不深入。本轮的授权范围是方案与最多 3 小时本机实测；下一轮 DFT 校准仍需单独授权。

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
| MACE-MP-0b2 small | 作者 GitHub release 权重已下载，MPTrj 训练，89 元素；本轮在 CPU/MPS 实跑 | **本轮选择的冻结 MLIP**；只是速度过关 | official_index/code，S9 |
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

### 2.1 起步体系 [inference；构造已 code 核验]

母相：立方萤石 ZrO2，2×2×2 常规胞，32 个阳离子位、64 个氧位。固定 4 Y、2 空位（每 2 个 Y 替换 Zr，移去 1 个 O 进行电荷补偿），实际 94 原子。Y2O3 摩尔分数为 `nY/(2*Ncation-nY)`，4/60=6.667%；**4/32=12.5% 是阳离子比例，不能混用**。

本机探针用常规胞 a=5.14 Å，即超胞 10.28 Å。这是运行测试的固定几何，不是已由 DFT 标定的晶格常数。正式比较统一使用校准后的共同晶格参数，首先只放松内部坐标。

不默认排除 Y–空位最近邻，不把 `<210>`、NNN 等文献答案硬编码为合法性条件。否则“学到了 NNN 偏好”会变成我们事先写死的结果。限制子空间仅作为与 Parkes 复现协议的独立附录。

### 2.2 动作与观察 [inference]

```json
{"op":"swap_cations","sites":[3,24]}
```

表示交换一个 Y 位和一个 Zr 位；例如实际编号以工具返回的 0–95 格点表为准，不固定上述示例对应元素。

```json
{"op":"move_vacancy","sites":[41,77]}
```

表示交换一个空位和一个 O 位。起步允许任意同子晶格两位交换；这是**优化动作，不是物理跳跃轨迹**，不能据此推导迁移时间或能垒。邻近跳跃可以作为统一动作空间的消融。非法 JSON、跨子晶格、同位交换不执行，并记入尝试数、token 和耗时。

每一步可见：晶胞、成分、格点编号/分数坐标、Y 和空位列表、当前与历史最佳的 MLIP 相对能量、最近 4 步动作/能差、已用预算、由同一工具计算的第一至第三配位壳统计。所有方法获得同样的信息；不向智能体泄露 DFT 终评或未来候选能量。若提供候选动作的能量预览，**每个预览也计调用**，不能算免费提示。

初版使用紧凑 JSON 状态而不是整份 CIF。相同状态的坐标轴变换、原点平移和格点重编号做训练增强与推理一致性测试，防止模型只背编号。

### 2.3 对称性与能量函数 [inference；去重已 code 核验]

- 在**无缺陷母晶格**上用 spglib 求空间群操作，并将它们转成格点排列；不是用已经降对称的缺陷结构求操作。
- 对全部等价占位串取字典序最小串作为构型键。本机得到 **1,536 个格点排列**；随机 100 构型的键全不相同，已检验若干平移/旋转得到相同键。脚本见 [lattice.py](scripts/lattice.py)、[test_lattice.py](scripts/test_lattice.py)。
- `X` 只表示 CE/占位表中的空位。送入 MLIP 或 VASP 前移除，保留真实原子的 `site_id`。
- 正式工具将占位先放到标准的母晶格、按构型键固定初始微扰，再做确定设置的 MLIP 内部弛豫；不把上一步残余坐标隐式带到新构型。否则“同一占位”会因不同历史落入不同局部极小值，缓存失效。
- 缓存键包含成分、胞参数、规范占位、模型 SHA、初始几何/微扰、弛豫参数。记录弛豫后是否跨位、是否可映回母晶格、最终几何哈希；离格/跨位构型单独报告并用结构匹配去重，不默默丢掉高分或低分样本。
- 训练/验证按对称性等价类以及轨迹连通的构型家族分组；同时隔离被留作终评的终态，不能只切开相邻两步做随机划分。

## 3. 冻结验证器与独立 VASP 终评

### 3.1 选择 [official_index/code；物理适用性待校准]

**选 MACE-MP-0b2 small**，公开 release、MPTrj 训练、固定 SHA256：

`d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9`

67,622,684 字节，来源和校验见 [manifest](models/manifest.json)。本轮选择它是因为已实际在本机运行，**不是已证明比 UMA 或专用 YSZ 势准确**。MACE 0b3 release 本轮列出的仅 medium；不虚构 small-0b3。UMA 和专用 G-NN 留作未来校准比较，不在这轮再下载所有大模型。

MACE **0.3.16** 的逐原子输出存在无条件 `.double()`，原版 MPS 报错。本机已在独立 `.venv` 内仅取消该辅助输出的双精度转换；总能量和力表达式未改，权重未改。CPU/MPS 均使用相同修正版本；MPS 载入前在 CPU 将权重转为 float32。修正：[patch](scripts/mace_mps_compat.patch)、[修改前源文件](sources/mace_models_original.py)、[哈希记录](sources/mace_patch_manifest.json)。原始失败保存在 `results/unpatched_mps_console.log`。正式复跑必须带上修正，不能只写“原版 MACE 原生支持”。

### 3.2 数据重叠和独立性的边界 [official_index/inference]

**已经核实**的是 MACE 官方声明的训练数据来源 MPTrj；**没有核实**训练库中全部 YSZ 占位与我们未来终评的重叠。完整训练包约 1.62 GB，本轮只保存 release 清单，未下载审计。不能因为是另一次 VASP 计算，就自动宣布终评从未见过。

正式终评前：冻结势权重和智能体版本 → 固定/封存新浓度与 Sc 任务 → 对候选占位/结构与可取得的势训练库做成分、对称键和结构近邻审计 → 删去精确重复或作为单独的 seen 子集 → 新做封存 DFT。化学体系相同不等于标签泄漏；同一占位的相关弛豫帧或相同参考能量已用于拟合，则不能作为严格未见构型成绩。若训练库审计做不全，报告“训练重叠未排除”，并用可追踪的新构型/专用势作补充。

校准集与共同 DFT 训练池可供所有方法使用；最终 DFT 终评集不参与势选择、SFT、筛轨迹、调参数或停止轮数。D0/R1–R4 的所有轮次可用预先抽定的**诊断集**观察代理偏差，诊断集另计，不混入最终盲评。循环 MLIP 保持冻结；更新评分器是另一个实验。

### 3.3 VASP 草案 [inference；依据 S1/S17]

| 项目 | 起步设置与校准 |
|---|---|
| 泛函、电子态 | PBE、YSZ 先非自旋极化；先用少量 spin 对照检查。YSZ 不默认加 U |
| PAW | 同一发行版的 `Zr_sv`、`Y_sv`、`O`，记录 TITEL、ENMAX、版本和哈希；不跨批混用；Sc 转移另用同发行版合适 Sc 势 |
| ENCUT | 起步 520 eV，并至少为所用 POTCAR 最大 ENMAX 的约 1.3 倍；3 个代表构型补 650 eV 对照，以相对能量收敛为准 |
| k 点 | 起步 Γ 中心 2×2×2；3 个构型补 3×3×3。Γ-only 仅做成本对照，不直接当终评设置 |
| 电子收敛 | `PREC=Accurate, EDIFF=1E-6, ISMEAR=0, SIGMA=0.05, LASPH=.TRUE., LREAL=.FALSE., ISYM=0`；检查绝缘性、SCF 收敛和设置敏感性 |
| 离子弛豫 | 固定共同晶胞，`ISIF=2, IBRION=2, NSW=200, EDIFFG=-0.02` 起步；关键结构加严到 0.01 eV/Å |
| 单点 | 同一 MLIP 弛豫几何上 `NSW=0`，测代理能量偏差和力；不能单凭单点证明构型已经稳定 |
| 主终评 | **DFT 内部弛豫后能量**为主；同组成/同晶胞内比相对能量。每个浓度独立参考，不能直接比较不同成分的裸总能量 |
| 收敛容差 | 目标是关键相对能量对 ENCUT/k 网格变动 <0.01–0.02 eV/超胞；若做不到，提高阈值或加严设置，不报告更小提升 |

校准出现明显反向排序、MACE 最优群在 DFT 中整体失去优势，或几何残余力明显异常时，先更换/校准评分器，不继续用它累积自训练数据；6 个弛豫的小样本只能发现严重问题，不能证明整个空间可靠。[inference]

主终评采用“单点诊断 + DFT 内部弛豫”，不只做单点。不同算法生成的终态使用完全相同的流程；不同局部极小值/收敛失败计入结果。对最优和几个边界构型做第二初始微扰检查，计入储备。此设计检验固定胞构型优化，不能据此宣称全相图、真实高温平衡或全局基态。

## 4. Baseline 与公平预算 [inference]

**两个横轴一起报告**：(1) 累计原始 MLIP 能量/力评估数；(2) 同一台机器上的累计 CPU/MPS 时间，并分开列训练、推理、搜索与预处理。一次弛豫可能含上百次评估，不能把一次弛豫和一次单点都记成“一次调用”。相同时间上限允许 CE 做更多便宜 MC 步；不能用固定 MC 次数人为削弱它。

| 方法 | 起步实现与公平约束 |
|---|---|
| 随机枚举/随机重启 | 同占位空间采样，母晶格对称去重，同 MLIP 弛豫；小空间可穷举时必须报告；共用合法性规则 |
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

## 5. 训练流程与每轮资源 [inference，未执行语言模型训练]

**基座**：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，关闭 thinking，只输出动作 JSON；LoRA rank 16、alpha 32、dropout 0.05，先 q/v 投影；上下文 1,024 token，batch 1、梯度累积 8、梯度检查点，学习率起步 1e-4。1.7B 是后续规模消融，不同时铺多个模型。使用 MPS 可用浮点方案；不依赖 CUDA/bitsandbytes。超参只在验证任务上改，终评不挑最好的轮次。

这里“RSI”采用用户的广义自我改进口径，算法名写成**迭代物理验证自训练**，不暗示已经获得自我改写能力、无限增长或比所有 RL 更强。

| 阶段 | 原始 MLIP E/F 评估硬预算 | 轨迹/样本设计 | 训练 seen-token 上限 | MPS 训练时间上限 |
|---|---:|---|---:|---:|
| D0 搜索示范 + SFT | 32,768 | SA/GA/随机探索混合，最多 32 条×16 操作；所有弛豫内部评估累计计数，预算耗尽即停 | 262,144 | 4 小时 |
| R1 自采样与训练 | 32,768 | 最多 32×16；保留高回报且构型不同轨迹，并保留一部分有用的升能探索 | 262,144 | 4 小时 |
| R2 | 32,768 | 同 R1，加入固定比例 D0/早轮回放 | 262,144 | 4 小时 |
| R3 | 32,768 | 同 R1 | 262,144 | 4 小时 |
| R4 | 32,768 | 同 R1，随后冻结最终模型 | 262,144 | 4 小时 |
| 单种子合计 | **163,840** | 轨迹数是上限，不能同时保证每次弛豫都完成 512 个动作 | **1,310,720** | **20 小时** |

每条训练记录包含状态、动作、工具结果与轨迹终回报；loss 只针对动作输出，**seen tokens 统计仍包含输入上下文**。记录 unique 数据 token 与实际重复训练 token 两种口径。按起点与回报分层抽样，轨迹重放按 25% D0、25% 早轮、50% 新轮起步，避免只复读一个已找到的排布。轨迹回报比较同起点的改善与最后的最好能量，不能把高能起点产生的大降幅误当作更好终态。

**时间解释**：本轮实测 MPS E/F 单点约 0.036 s。32,768 次约 20 分钟纯评估只是测得吞吐的线性外推，未包含全部几何差异、智能体推理和文件开销。以当前一条弛豫约 106 步估算，每轮可能只够约 300 次完整候选弛豫。将最多 512 次动作理解成必达数会超预算。

**语言模型训练速度尚未测量**；262,144 seen tokens 若有效吞吐为 20–100 tokens/s，对应约 3.64–0.73 小时，这是敏感性情景，不是 M3 Max 实测。正式训练先测 100 步吞吐；4 小时先到就停止，所有 LLM 对照匹配实际 token/步数/时间，不能把没训练完的轮次当完整同 token 结果。推理时间与训练时间分开，MACE 和语言模型顺序使用 MPS，避免同时抢显存带宽污染计时。

至少做：不更新权重、只用 D0、同最终数据一次训练、搜索蒸馏、非语言小策略、对称增强开/关、回放开/关。完整矩阵只在代理指标有重复性后展开，DFT 配额优先主方法与最强对手。

## 6. 分级迁移 [inference]

| 等级 | 任务 | 本轮决定 |
|---|---|---|
| L0 | 同 6.667% 成分/晶胞，新随机起点、格点编号与坐标变换 | 需要；测试搜索和不变性，**不算新材料泛化** |
| L1 | 新浓度：同 32 阳离子位，6Y+3空位＝Zr26Y6O61（93 原子），10.345 mol% | **主 DFT 终评必须含这级**；不在此浓度训练/调参，目标成分单独能量参考 |
| L2 | Sc 替换 Y：Zr28Sc4O62 | 先做独立 Sc 评分器校准；通过才纳入小配额 DFT 终评，否则只能报告 YSZ 跨浓度迁移，不声称跨元素 |
| L3 | 新基体：如 Y 掺杂 CeO2 | 本研究首轮排除；Ce3+/4f 局域态、U、自旋与缺氧机制须重建协议，不能沿用 YSZ 的“非磁简单”假设 |
| L4 | 更大晶胞/不同胞形 | 后续可选；3×3×3 成本先实测，不承诺塞进当前主预算 |

同成分换起点的能量最小值是同一个目标，模型可能只背答案。论文若要讲迁移，至少需要 L1；若 L1 不成立，收缩为单景观策略学习。对主终评使用冻结的 R4、相同 SFT 对照和最强搜索对照，不能先看 DFT 再选择轮次或目标浓度。

## 7. 材料问题与可支持的解释

**[direct_text; S1/S2/S3/S7]** 文献对 Y–空位 NNN、空位对方向、有序聚集及电导的讨论高度依赖浓度、约束、胞形与温度。S1 的 `<210>` 是固定小胞、6.7% 和受限候选中的低能结论，不应当作全部 YSZ 的普适常数。

**计划统计 [inference]**：按母晶格配位壳，比较 Y–空位 NN/NNN/第三壳、Y–Y、空位–空位的数量/条件概率；对相同组成随机排布作富集比参考。空位对用周期性最小像，并记录等距离多重像，按对称方向族归类，避免任意选一个等长向量。分别画 MLIP 入选集与 DFT 后低能集，检查偏好随训练轮数是否只在代理上增强。报告不同种子和独立构型家族的不确定度；从搜索样本频率不能直接推导热平衡概率。

**若声称提高离子电导，至少另需 [inference]**：代表性 O 跳跃的 DFT NEB/势能垒核验；足够连通的跃迁网络与 kMC，或校准过的长时间多温度 MD；扩散统计、相关运动/电导换算、有限尺寸和采样收敛。低构型能量、某条短 NEB 或“空位更分散”都不等于电导更高。本轮主论文先不承诺电导提升。

## 8. 50,000 核时预算与判据 [inference，尚未提交]

核时＝分配 CPU 核数×实际运行小时；排队时间另记。以下是封顶分配和成本情景，不是集群实测。

| 部分 | 核时上限 | 做什么 |
|---|---:|---|
| 校准 | **5,000** | VASP 截断能/k 点/晶格/自旋小检查、MLIP 能量排序/力偏差、少量 Sc 检查；**下一轮先申请其中 3,000** |
| 探针 | **7,500** | 共同 DFT 训练池、最强搜索 vs D0/早轮的小比较；CE–DFT 起步目标约 40 个总弛豫标签（含可复用校准标签），实际数量按实测核时调整 |
| 主实验 | **27,500** | 等额候选配额的独立终评，优先 L0/L1；目标约 80 个去重后终态起步，L2 通过校准才给配额；关键设置复核 |
| 储备 | **10,000** | 重算失败、收敛加严、第二初始几何、CE 增样不足、有限的 Sc 复核；未触发则不花 |
| **合计** | **50,000** | 本轮实际使用 **0 DFT 核时** |

### 8.1 成本情景与第一批申请

以 32 核作业举例，94 原子、PBE、520 eV、2×2×2 k 网格的单点暂按 **8–32 核时/构型**（0.25–1 h 墙钟），内部弛豫按 **80–400 核时/构型**（2.5–12.5 h）排预算。这些只是足够宽的工作估算；实际节点、VASP 编译、赝势、SCF/离子步和 k 点可显著改变它。S2 的 324 格点约 10^4 CPUh/全弛豫是旧硬件和不同设置，不能直接当我们的成本。

**下一轮具体授权提案：首批封顶 3,000 核时**，包含：

1. 12 个 6.667% YSZ 构型做单点：4 随机、4 MACE 低能、4 点电荷/MLIP 分歧或 NN/NNN 边界构型。
2. 从中每组取 2 个，共 6 个做固定胞 DFT 内部弛豫。
3. 3 个代表构型各补两个收敛单点，共 6 个：650 eV 与 3×3×3 网格分别对照基准设置。
4. 先提交 2 个代表任务测实际成本，再在剩余额度内完成其余；累计申请/预计消耗超 3,000 即停止扩批。本轮没有生成提交命令或连接集群。

这批最多 18 个单点+6 个弛豫。按上述上沿 18×32+6×400=**2,976 核时**，可作为请求上限，但单任务若更贵必须减量，不能突破封顶。它能回答运行成本和粗略排序是否可用，**不能凭 6 个弛豫就宣布势全面准确**。

校准后用实际 `c_SP`、`c_relax` 替换所有情景：每阶段数量满足 `N_SP*c_SP + N_relax*c_relax + 复算成本 <= 阶段额度`。主实验的 80 个终态是规划目标，不是已保证可完成。若成本高导致 CE 标签或终评不足，使用储备、缩小目标问题或报告无法判断，不能只留下弱 CE。

### 8.2 支持 / 不支持 / 无法判断

预先登记实用能差阈值 `δ=max(0.05 eV/超胞, 3×数值收敛误差)`，仅用于同成分比较；阈值在正式盲评前锁定。

- **支持 [inference]**：RSI 相对最强搜索与同最终数据一次 SFT，在相同累计时间下达到相同低能档成本至少降约 20%，或等额终态配额的 DFT 弛豫能量改善超过 δ；至少 3 种子方向一致，配对区间支持增益；并在 L1 同样保留效果，不以多样性彻底丢失换来。这里是支持开展完整实验的预设标准，不是已有结果。
- **不支持 [inference]**：有足够、收敛一致的独立 DFT 标签，最强搜索/CE/同数据一次 SFT 已与 RSI 实用等价，或 RSI 更差；或者提升仅发生在代理分数、DFT 不跟随。收缩主张、研究训练收益边界/代理偏差，或停止这一算法路线，不继续凭轮数叠计算。
- **无法判断 [inference]**：终评标签太少/区间跨零且跨实用阈值、势训练重叠无法排除、不同设置误差大于提升、弛豫大量跨位或未收敛、CE 因数据不足而明显欠拟合。先解决诊断问题，不能把统计不显著写成证实无效，也不能把代理上分写成成功。

DFT 终评按方法、种子和任务预先等额选候选，取并集后统一盲算；重合候选复用同一标签，选择性失败和重复率也报告。比较对象是各方法实际找到的最好已知结果，不声称整个占位空间的全局最优。

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

## 10. 下一轮执行顺序与授权

1. **先审这份方案，锁定主指标、训练/终评边界和公平预算**；不是重开方向选择。
2. **需用户下一轮明确授权：VASP 首批最多 3,000 核时校准**，具体任务见 §8.1；还需确认实际集群入口、队列、计费核时规则、VASP/PAW 可用版本。可先准备输入与作业清单，不提交。
3. 校准没有明显排序失效后，再授权小模型 D0+R1/R2 训练探针；本轮未下载 Qwen 权重、未启动 SFT/RSI。先实测语言模型吞吐，再按 §5 的实际成本匹配对照。
4. 探针有可重复收益，才展开 R3/R4 和完整封存 DFT 终评；Sc 和输运都不自动扩张预算。

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
- **S17 [official_index]** VASP 官方 [ENCUT](https://vasp.at/wiki/ENCUT)、[PAW 目录](https://vasp.at/wiki/Available_PAW_potentials)；只支持参数定义与需收敛测试，不支持我们尚未测得的核时估算。
- **S18 [official_index]** [icet 官方文档](https://icet.materialsmodeling.org/)、[Qwen3-0.6B 官方卡](https://huggingface.co/Qwen/Qwen3-0.6B)；icet 已实际跑通，语言模型未训练。
