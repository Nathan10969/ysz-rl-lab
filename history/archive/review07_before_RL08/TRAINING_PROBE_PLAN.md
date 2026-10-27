# 训练探针计划：先测吞吐，再审 D0 + R1/R2

日期：2026-10-23。模式：当前冒烟执行口径与正式探针设计。**当前仅有准备/预检记录，GPU冒烟测量待DFT前驱结束后运行；本文修订不运行或提交作业。**

依据：[REVIEW_CLAUDE_07_gates.md](REVIEW_CLAUDE_07_gates.md)、[PLAN.md](PLAN.md)。环境合同见 [TRAINING_ENV_SPEC.md](TRAINING_ENV_SPEC.md)。`direct_text`＝来源正文明确陈述；`code`＝已读代码/已有原始文件；`official_index`＝官方目录；`inference`＝本计划设计或估算。下文未另标的设计、阈值、数量与预算均为`inference`，**A100训练/MACE速度全部待测**。

## 1. 已授权与拟议范围

| 范围 | 状态与界限 |
|---|---|
| 训练冒烟池 | 用户已批10 GPU卡时，独立于40卡时DFT池；当前两个单卡作业硬预留**0.50+0.75=1.25卡时**，其余**8.75卡时未分配，不自动使用**。仅0.6B tinySFT、316原子float64 MACE单点/有界弛豫测速，无额外规模测试、1.7B或RSI |
| 正式小探针 | 本文提出D0+R1/R2、3个固定种子，先做§7的四类最小探针；**尚未批准额度，不执行**。九arm为扩展，不是唯一启动选项；不沿用旧PLAN中D0+R1–R4的大表作为本轮授权 |
| 正式DFT诊断/共同训练池/盲终评 | 单独拟议和记账；不占用10卡时冒烟池，不从现有40卡时校准池借支 |
| 1.7B、R3/R4、5–10种子、L1迁移 | 后续设计；不属于本10卡时冒烟范围或本次三轮小探针的硬预算 |

M3 Max只做本机开发；A100 **40GB**单卡作业，每job固定**1 GPU、6 CPU、90 GiB主存**。平台账号实际最多**2 GPU作业并发**，不是8；这是全账号上限，DFT与训练同时活跃也不能合计超过2。主存90 GiB不等于GPU显存40GB。[审核07、用户环境；code/direct_text]

## 2. 要回答的问题与证据

问题是：在同一冻结评分器、同累计可用计算资源下，分轮更新策略是否比强搜索、冻结模型继续采样、一次性训练更容易找出独立DFT认可的低能末态。仅证明模型能输出JSON、SFT loss下降或MACE能量变低，都不回答这个问题。

| 判断 | 对应证据 | 本探针可支持的边界 |
|---|---|---|
| 工具链可用 | tinySFT checkpoint保存/重载、合法JSON、有界MACE弛豫状态/原始E-F计数、真实吞吐 | 冒烟只判断工程可用性；40步未收敛不算有效评分 |
| 分轮更新有效 | D0→R1→R2的配对曲线，和冻结/一次性同数据/搜索蒸馏比较 | 同一316原子任务的小规模方向性证据 |
| 物理收益保留 | 等额top-k盲DFT低能率、每轮独立诊断DFT | 不以代理提升替代DFT |
| 可迁移 | L1新浓度独立终评 | 本小探针未覆盖，不声称跨材料/浓度泛化 |

审核07放行的是当前校准池里的MACE评分器。约0.195 eV/胞的校准相对能RMSE不是新生成分布的误差界。[审核07；code] `δ_proxy=max(0.30 eV/胞,3×数值相对能误差)`为代理赛道实用门槛；低于它不能解释成可信的物理提升，高于它也不能免除DFT。[inference]

## 3. 当前冒烟包：两个作业共预留1.25卡时

以 [training_smoke/README.md](training_smoke/README.md)、[JOB_MANIFEST.json](training_smoke/JOB_MANIFEST.json)、[BUDGET.json](training_smoke/BUDGET.json) 为当前唯一冒烟清单。两个作业均为1 GPU、6CPU、90GiB主存；当前硬预留1.25卡时，**10卡时授权池剩余8.75未分配，不是纠错储备或自动扩批额度**。[code]

| ID | 当前固定工作负载 | Slurm硬封顶 | 输出与当前状态 |
|---|---|---:|---|
| `c07_smoke_sft` | Qwen3-0.6B LoRA rank16/alpha32/dropout.05、q/v；64条训练+4条验证，batch1/累积2，最多64微批/32优化步；序列上限512，seen-token硬上限32768；4条greedy验证生成各最多64新token，合计≤256 | 0.50卡时（30分钟） | 预检已确认本数据**计划16,023个非padding seen-token、单条最长258**；这是分词统计，不是已完成训练。GPU loss/梯度/显存/checkpoint/生成合法性与短样本吞吐仍待测；不含独立prefill/decode长度扫测 |
| `c07_smoke_mace` | 冻结316原子MACE float64；8个独立随机输入胞；单点batch1/2/4，各2次预热+10次稳态，含batch2/4与对应单胞E/F一致性检查；固定胞FIRE弛豫6胞（2个单胞+1批4胞），每胞≤40步，fmax目标.05 eV/Å；全部原始E/F≤1024 | 0.75卡时（45分钟） | 冷加载、构图/传输、前向/结果传输、端到端p50/p95、原始E/F与内存/CPU成本；弛豫收敛、跨位/末态键和实际卡时均待测 |

两job的Python内部工作限时分别900/1800秒，外层timeout分别1500/2400秒；实际账按Slurm持卡时间，不能用Python计时替代账单。tiny数据是提示中直接给出两个候选位点的**无评分合法动作示例**，生成不调用MLIP，不含DFT标签，只教JSON格式和SFT流程，不称D0、优质轨迹或搜索能力结果。MACE输入从固定种子独立生成，只读取几何/组成，不读取校准或终评DFT标签。[code；数据/预检清单]

**排队顺序：DFT先行。** GPU DFT八项已提交，ID为617、619、621、623、625、627、629、631；CPU补跑已提交1890508，见各自[GPU提交账](calibration_gpu/claude07/SUBMISSIONS.json)、[CPU提交账](calibration/claude07/SUBMISSIONS.json)。两个冒烟安排在全部GPU DFT前驱终态之后；提交依赖须覆盖八项（已核实两条资源链尾629/631覆盖全部八项时可用相应`afterany`）。具体冒烟job ID/状态由父任务提交账记录；目前本文没有GPU冒烟测量结果。[code；父任务执行状态]

全部DFT前驱结束、两个GPU槽位均空闲时，两job并行跑满的计算墙钟为45分钟；若仅一个槽位可用而串行，合计最多75分钟。排队/DFT等待另加，**不是从当前时刻算的完成时间预测**。无1.7B运行，无512/1024/2048扩展测速job，无float32或batch8/16扩批job；不因未用完10卡时而追加任务。失败先定位，不自动重试，独立失败不连带取消DFT。[code/inference]

40步FIRE达到步数上限但未达fmax时标记`step_capped_unconverged`，保留末态/成本，不能当成收敛训练标签。当前实际最长258token的短样本吞吐不是2048token训练测速；本次FIRE/.05/40步也不是正式BFGS/.03/300步的完整弛豫测速。两者能填基础工程参数，但正式60/162卡时预算的可完成数据量仍属暂估，不自动产生进一步尺度测试。[code/inference]

## 4. 基座与训练配置

| 选择 | 决定和依据 |
|---|---|
| Qwen/Qwen3-0.6B | 当前主基座。官方卡标0.6B、28层，支持关闭thinking；较小模型有利于先分辨流程/数据是否成立。没有证据说明其在YSZ上优于1.7B；更快/更省显存是待实测的设计预期。[direct_text Q1；inference] |
| Qwen/Qwen3-1.7B | 官方卡标1.7B、28层；仅后续规模对照。只有0.6B流程通过且验证集显示持续的动作理解/格式瓶颈、扩大容量可能有价值，才独立设计并计量。不能按参数比线性外推吞吐或预设一定更好。[direct_text Q2；inference] |
| 加载/精度 | 0.6B原始权重BF16，LoRA适配器训练；单GPU普通Transformers/PEFT，先不用4bit、DDP、ZeRO或FlashAttention额外依赖。BF16/具体attention路径及显存可用性要在冒烟登记；遇不兼容先定位，不静默换协议。[inference] |
| LoRA | rank16、alpha32、dropout.05、bias=none、target_modules=`q_proj,v_proj`、task_type=CAUSAL_LM；基座/embedding/lm_head冻结；启动时打印实际匹配模块、trainable names/数量，做一次梯度检查。参数是本项目起步设计，不是官方YSZ推荐。[inference；Q4支持API语义] |
| 正式SFT拟议 | 序列≤2048，microbatch1、梯度累积8、梯度检查点；AdamW起步lr=1e-4、weight_decay=.01、clip1.0，warmup为本阶段计划步数的10%；只监督动作与结束符。当前tiny包实际为累积2、序列上限512，按§3执行，不当作正式配置已测。[inference/code] |
| 轮间状态 | 同一LoRA从D0续到R1/R2，不每轮重置适配器；每阶段重置optimizer并按该阶段预算重建scheduler，保证一次性对照可清楚匹配累计步/token。检查点选阶段最后一个完整有效更新，不按盲DFT择优。[inference] |
| 推理 | 官方chat template，hard `enable_thinking=False`，每动作新token≤64，temperature.7/top_p.8/top_k20；同版本模板适用于SFT和推理，详见环境规格。[direct_text Q1/Q3；inference] |

官方卡指出Transformers<4.51.0会遇到`qwen3`识别错误；这只是兼容下限，不是本项目已验证的完整环境。运行前固定模型revision/tokenizer/template SHA和实际 `torch/CUDA/transformers/peft/accelerate`锁；MACE环境与LLM环境可分开，避免升级现有DFT环境。[direct_text Q1；inference]

当前远端准备记录已确认：LLM为torch2.7.1+cu128、transformers4.57.1、peft0.17.1、accelerate1.10.1；MACE为torch2.7.1+cu128、mace-torch0.3.16、ASE3.29.0、spglib2.7.0。Qwen权重从官方ModelScope取得并按逐文件revision/长度/SHA核对，HF原始网络在集群不可达；共享LLM/MACE环境只读复用，spglib在项目overlay安装。以上是准备/导入/tokenizer预检通过，**不代表GPU模型前向、训练或测速已经通过**。[code；PREPARATION_STATUS与sft_preflight]

已核验的关键指纹：Qwen `model.safetensors` SHA256=`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`；`tokenizer_config.json`（含chat template）SHA256=`d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`。逐文件revision、完整tokenizer与长度校验见 [qwen_manifest.json](training_smoke/evidence/qwen_manifest.json)，不能只引用可变的网页main。此核验在CPU/文件阶段完成，没有替代GPU训练结果。[code]

Qwen官网的LLaMA-Factory页目前明确写着“待为Qwen3更新”，示例面向Qwen2.5；因此不能把那份示例直接当成Qwen3 LoRA已跑通的证据。LoRA的API语义以PEFT官方文档为准，具体模块匹配以冒烟日志为准。[direct_text Q5/Q4]

## 5. 数据边界、D0来源与三固定种子

训练/搜索主种子固定为 **2026102301、2026102302、2026102303**。用带固定命名空间的确定性派生流区分`teacher/train/validation/diagnostic/blind/augmentation/decode`，保存派生算法与清单SHA；不能靠同一个种子不断顺序取数，把划分依赖执行顺序。三个种子构成三个独立训练/搜索运行，不能把同轨迹的相邻状态当独立重复。[inference]

| 数据池 | 来源与用途 | 明确隔离 |
|---|---|---|
| D0训练 | 主任务合法随机起点；同冻结MACE/同可见信息的SA、GA、CE–MLIP+MC和随机重启搜索轨迹。先记录全部尝试再按终回报及末态家族筛选 | 不读保留DFT标签；不把Parkes论文的2,857构型当已持有数据 |
| 验证 | 独立起点/完整轨迹家族；仅用于搜索参数、编码、训练配置的预登记选择。小探针内搜索调参在D0额度内封顶，不看盲DFT | 数据增强与重编号不跨家族；冻结正式配置后不按终评调整 |
| 每轮诊断 | RSI每seed每阶段4份未进入训练的末态：2低能尾部、1按规范键哈希随机、1配位/几何多样性样本；静态对照只做同规则终点审计，去重不足则如实报告 | DFT标签仅用于代理偏差监控；不进入SFT、CE、轨迹排序/选老师；诊断起点/家族与盲终评分开 |
| 盲终评 | R2最终checkpoint冻结后，8个封存起点×最多8动作；各方法同起点/预算。只返回冻结MACE反馈，最终候选DFT由独立评估端封存 | 这些轨迹/末态不回灌训练、验证或早停。发现训练家族碰撞计入不足配额失败 |
| DFT共同参考池 | 正式训练前封存的主胞参考结构清单与同协议DFT能量；评估端固定低能阈值 | 不能用参赛方法最终最优结果倒推阈值；代理赛道不向老师/模型暴露标签 |

D0不使用校准DFT分数挑“好示范”，也不从盲标签提炼规则。校准的作用仅是放行冻结评分器。保留DFT包括校准后专门封存的诊断/盲评标签；若后续开展“共同DFT训练池”赛道，需给全部方法同标签并另立实验，不能混进本代理赛道。[inference]

D0每seed最多32条8步教师轨迹，四类老师各最多8条、原始E/F各暂分8192，合计32768；CE标签启动/拟合/调参/最终回MACE复核都在其份额内，不能免费多拿标签。每老师预留5个训练起点、2个验证起点、1个诊断起点，合计20/8/4条；诊断4条的全部家族不入训练或验证。R1的16条预分12训练、4诊断；R2的8条新采集预分4训练、4诊断，另8条才是冻结后的盲评。不足时先报告，不由其他老师或盲数据补齐。起点弛豫也占E/F额度，因此“32×8”是尝试上限而非保证完成数。[inference]

SA/GA参数只在固定验证轨迹选择；CE依照同一个初态→冻结弛豫末态能量目标拟合，必须报告留出误差和低能尾部偏差，不能以理想格点CE smoke冒充弛豫景观拟合。高回报轨迹按末态规范家族和起点分组，保留有意义的升能中间步；每起点最多2条入选，避免一条幸运搜索占满D0。[inference]

可复核的起步搜索设置：SA按末态能差接受，`P_accept=min(1,exp(-ΔE/T))`，T是算法的能量尺度，默认0.30→0.01 eV几何下降，不解释成材料温度；GA种群8、成分守恒均匀交叉、每子代一次合法交换变异、末态键去重后保留精英；CE起步pair/triplet cutoff=5.2/3.0 Å、ridge，λ只在`1e-6/1e-3/1e-1`中用固定验证集选，MC最终候选回同MACE协议。拟合/修复/调参/初始种群/预测筛选后的每个真实评分都计成本；验证不足则称“起步配置未充分调优”，不能宣称已穷尽强搜索。三个老师的独立完整搜索arm在扩展矩阵中用于复核预选结论。[inference]

R1由D0模型自己提出动作，R2由R1模型提出动作。R1回放50%D0+50%新轮，R2为25%D0+25%R1+50%R2；比例按实际训练的非padding seen-token计，若家族独立有效样本不足，缩小训练量并报短缺，不能重复同一末态充作新数据。所有筛选依据冻结MACE末态、合法性、多样性和完整轨迹回报，不按DFT诊断标签筛选。[inference]

## 6. 正式小探针：每阶段的硬封顶

本表将PLAN原D0+R1–R4的大规划收缩到**D0+R1/R2、0.6B、3seed**；这是待审的独立小规模额度。计数定义见环境规格；每项上限先到即停，不假定能跑满目标样本。

| 每seed阶段 | 原始MLIP E/F | 数据动作尝试上限 | 新生成token硬封顶 | rollout输入/prefill token硬封顶 | 训练seen-token | 优化步 | LLM池/MLIP池/共同卡时上限 |
|---|---:|---|---:|---:|---:|---:|---|
| D0 | 32,768 | 教师32×8；另≤64次格式验证生成 | 4,096 | 126,976 | 262,144 | 32 | 1 / 1 / 2 |
| R1 | 16,384 | 自采样16×8；另≤64次格式验证生成 | 12,288 | 380,928 | 262,144 | 32 | 1 / 1 / 2 |
| R2 | 16,384 | 自采样8×8 + 最终盲起点8×8；另≤64次格式验证生成 | 12,288 | 380,928 | 262,144 | 32 | 1 / 1 / 2 |
| 每seed合计 | **65,536** | 最多512个数据动作尝试；格式验证另192次 | **28,672** | **888,832** | **786,432** | **96** | **3 / 3 / 6** |
| 3seed单arm | **196,608** | 均按独立运行报告 | **86,016** | **2,666,496** | **2,359,296** | **288** | **9 / 9 / 18** |

R2的16384原始E/F预分8192给新数据采集、8192给冻结后的最终盲rollout；不能为多训练挪走盲评份额。D0/R1的起点、合法性验证的实际打分、轮间固定验证结构及最终复核均占各自份额。格式验证只查JSON，不另做免费物理搜索。诊断DFT取样复用已计费的MACE末态；追加MACE评估仍从阶段E/F中扣。[inference]

`seen-token`包含每次训练的prompt、模板和动作的非padding输入，重复epoch/回放每次重算；同时另报有损失的target token、唯一轨迹数、唯一末态家族数。padding分配量另报。rollout生成token包含非法JSON、截断和验证输出；prefill包含反复送入的上下文。不能把“唯一训练集只有几万token”当作实际训练只看了几万token。[inference]

LLM池1h包括模型冷加载/预热、数据等待时的GPU占用、采样prefill/decode、SFT、checkpoint保存/重载与验证；其中纯优化最多0.75h，其他至少预留0.25h，实际未用不转为额外优化。MLIP池1h包括加载、预热、单点/弛豫、计数/映射/去重和持卡空闲；不得把数据采集成本藏在训练前。每job仍按单卡≤2h封顶，阶段之间通过可审checkpoint衔接，不开常驻闲置服务。[inference]

LLM池和MLIP池是**记账类别，不是必须同时占两张卡的服务**。两卡排程默认每seed同一时刻只占1GPU，在该卡上依次做LLM提案与MACE弛豫；模型能否同驻显存待测，不满足时卸载/重载的全部成本进入对应类别。持卡时间按不重叠区间归类，两个模型同时驻留不重复算两卡，也不能把空等时间删掉。另一个账号卡位留给另一个独立seed/job。[inference]

## 7. 先四类最小探针，再扩展完整对照

最小启动包只做四类、各3seed：RSI、冻结D0模型继续多采样、同数据一次SFT，以及在SA/GA/CE中预选的最强搜索。最强搜索在D0固定验证轨迹上按“同累计资源下末态最佳能量、并列按独特有效末态数”预选，所有候选老师的验证成本已经在D0内，不看盲DFT。必须同时保留SA/GA/CE的验证结果，不挑最弱者。[inference]

| 最小arm | 每seed实际新增GPU上限与工作负载 | 3seed新增上限 | 成本归因 |
|---|---|---:|---|
| RSI D0+R1/R2 | 6h：LLM≤3h（纯优化≤2.25h）+MLIP采集/评估≤3h | 18卡时 | 含D0教师采集、全部更新/采样、最终盲rollout |
| 冻结D0多采样 | 共享RSI同seed的D0教师数据/checkpoint后，再用R1/R2两阶段各≤2h做采样/评分，无新增优化 | 12卡时 | 独立复跑仍归因D0≤2h/seed；可比总额度6h/seed，新增执行仅4h |
| 同数据一次SFT | 从原始基座训练；新增LLM≤3h（纯优化≤2.25h）+终点评估MACE≤1h、raw E/F≤16384（8192盲、8192诊断/复核）；盲8×8、诊断4×8与≤64次格式验证，输出≤10240、prefill≤317440 | 12卡时 | 不重做已完成RSI数据采集，但完整归因产生该数据所依赖的RSI执行日志。归因总成本可高于RSI，不能伪装成独立在线6h策略 |
| 预选最强搜索 | ≤6h均可用于MLIP/搜索，无LLM训练；raw E/F≤65536，最终盲配额先预留 | 18卡时 | 便宜CE/MC可在同时间内继续搜；不强造R轮 |
| **最小新增物理执行上界** | 数据/checkpoint共享与新增训练分别记账 | **60卡时** | DFT诊断、参考池和终评另计；非授权 |

四类中，RSI、冻结多采样与最强搜索的公平时间比较包括其全部采集、训练/推理成本；同数据一次SFT是**固定数据的机制控制**，主看相同seen-token/优化步/数据下分轮是否仍有优势。它的数据依赖来自RSI，不能同时声称它“独立从零只有4h/seed”或把其条件控制成绩称为等成本在线胜负。记录 `H_same_attributed = H_RSI_data_dependency + H_same_training_and_final_eval`，前项包括产生采样策略必需的更新/加载，按实际日志拆分；保守上界≤18+12=30卡时，实际账不重复付这18。[inference]

最小包的实际总账是60卡时**新增执行上限**，不是保证消耗：冻结arm复用D0、同数据arm复用RSI数据，其训练/采样工作负载不同；用实测后分别求时，不假设它们各花18。若需独立从零复现，再逐arm报告共享前的归因成本。[inference]

扩展完整矩阵补齐以下九类。独立采集一次SFT与搜索蒸馏是进一步辨别“谁采数据、何时训练”的必要控制，但不要求在启动四类探针时一起开跑。

| arm | 数据/更新与要排除的解释 |
|---|---|
| RSI | D0后用本模型采集R1/R2并更新；测试分轮更新本身 |
| SA | 同动作空间、同冻结MACE；允许升能与重启，含参数选择成本 |
| GA | 成分守恒交叉/修复/变异；所有修复、重复和打分计费 |
| CE–MLIP+MC | 同预算MLIP标签、icet多子晶格CE、廉价MC可做更多步，最终候选同MACE协议；报告标签采集/拟合/MC/回核成本 |
| 冻结D0模型继续多采样 | D0 checkpoint固定；把RSI用于更新的时间用于更多采样/搜索，不能闲置该时间 |
| 一次性SFT–独立采集 | 从公共D0教师数据起步，后续由未更新的初始基座按自己的种子流提出动作，按同累计采集资源收集，全部采完后从基座训练一次；不能免费使用RSI轨迹或先训练D0 checkpoint再称总共只训练一次 |
| 一次性SFT–同数据 | 从同初始基座用RSI最终累计可训练池一次训练，匹配实际seen-token及优化步；采集成本按完整RSI记录归入。是事后同数据控制，不宣称在线策略 |
| 一次性搜索蒸馏 | SA/GA/CE强搜索持续采集同预算示范，一次训练；排除“好老师加一次压缩”解释 |
| 分轮搜索蒸馏 | 每轮刷新SA/GA/CE示范并训练，模型不负责采集；排除“只是老师随轮更新”解释 |

在线arm每seed总GPU执行上限6h、原始E/F上限65536、相同最终盲rollout配额；同数据arm按上表新增执行≤4h、raw E/F≤16384，完整数据依赖另外归因。LLM可训练arm累计seen-token≤786432、优化步≤96。一次性SFT的这些上限一次使用；匹配对照按RSI**实际**达到的token/步数下调，不能只比较名义上限。[inference]

**资源允许重分配，但总额不可突破：**表6的1h LLM+1h MLIP是RSI工作负载拆分。纯搜索无LLM，可把每阶段2h全部用于MLIP/搜索；冻结模型可把无需更新的LLM时间用于额外生成。因此纯搜索每seed的MLIP池上限可达6h，冻结模型新生成token每seed另封131072、prefill≤4063232、动作尝试≤2048（含与RSI等额的最终64个盲动作）；原始E/F65536与共同GPU6h仍是共同硬闸门，实际先到即停。该差异用于让对手实际使用节约的训练时间，并完整披露。[inference]

主报告同时画(1)能量/成功数对累计原始E/F、(2)对累计实际总GPU卡时，附CPU核时。CE的便宜MC可在同时间内做更多步；不限制成与LLM相同MC步数。相同E/F截止点的曲线避免用更多物理标签解释收益，相同总时间曲线包括LLM训练代价。不能挑其中较有利的横轴作为唯一结论。[inference]

九arm全部执行的宽松上界为 **9arm×3seed×6h=162 GPU卡时**，不是当前授权或最小启动要求；同数据arm实际新增上限更小，共享D0还会减少实际值。这个数是全矩阵GPU执行封顶，**不是各方法完全独立从零取得RSI数据的归因成本总和**。LLM分配与MLIP分配属于同一162共同上限，不能把各自极端上限再相加。D0教师采集可按seed共享一次；RSI、冻结D0模型与分轮搜索蒸馏可共享同seed的D0 checkpoint，但一次性训练arm必须从同一原始基座开始。物理账/实际训练账只付一次，同时列每arm独立从零复跑成本与共享摊销。[inference]

独立采集的一次性arm可把三阶段预留的SFT预算累计到数据采完后使用：累计纯优化≤2.25h、LLM总额≤3h、共同≤6h、seen-token≤786432、优化步≤96；分成≤2h单卡job保存/续接。同数据控制则按最小包上表执行。各arm最终64个盲动作与8192原始E/F同样先预留，不得被一次性训练或独立采集挤占。非RSI的普通LLM采集arm采用表6累计生成/prefill上限；冻结多采样采用前述放宽上限。[inference]

随机重启/点电荷Ewald是D0和搜索流程的必要sanity子项，报告各自轨迹/成本；若要把它们列为两个独立完整arm，再增**36卡时上界**，不隐藏进162。非语言小策略、CE–DFT、abICS型主动学习和增强/回放消融仍是完整论文证据包的后续项，不在本小探针下宣称已完成强基线全覆盖；CE–DFT尤其需要另批共同DFT训练池。[PLAN；inference]

## 8. 两卡下的卡时、墙钟与吞吐替换公式

| 工作包 | 共同GPU硬上限 | 两卡理想下界 | 简单可实现的限额排程估计（无排队） |
|---|---:|---:|---:|
| 当前两个冒烟job | 1.25卡时 | 0.75h（最长job约束） | 全部DFT前驱结束后，两槽位空闲则≤45分钟；单槽串行≤75分钟；DFT等待/排队另加 |
| RSI：D0+R1/R2，3seed | 18卡时 | 9h | 两个seed并行跑3阶段共6h，再第三seed6h，≤12h |
| 四类最小探针，3seed，共享按§7入账 | 60卡时 | 30h | RSI与搜索按seed两列共18h，其后冻结与同数据按seed两列共12h，≤30h；DFT监控等待另加 |
| 九arm、3seed扩展执行封顶 | 162卡时 | 81h | 27条每条≤6h的依赖链两列排程，按独立链宽估≤84h；共享D0可减少实际值 |

这些是**硬封顶与确定排程的估计**，不是已测速度，也不是平台排队SLA。小任务若提前完成会缩短；DFT占卡、存储/上传、软件安装和等待人工诊断会增加总历时。只有无其他账号作业且两个槽位都可用时才用除2。若每job申请满6CPU，则18卡时对应≤108配套CPU核时、最小60对应≤360、扩展162对应≤972；平台如单独收费应另列，不折算为GPU卡时。[inference]

冒烟之后只把下表中**与当前短样本/有界FIRE协议匹配**的`TBD`填为A100实测；不把M3 Max秒数折算过来，也不把258token/FIRE40步结果冒充2048token/BFGS完整弛豫实测。正式预算仍为未批准、待对应吞吐校正的暂估。

| 符号 | 实测含义 | 当前值 |
|---|---|---|
| `L_LLM,L_MACE` | 每次冷加载+预热秒数；每次job重载各算一次 | TBD |
| `q_train(l)` | 当前tiny样本最长258、上限512，实际非padding seen-token/秒，含前后向/优化；正式2048token的`q_train`另列未测 | 短样本TBD；2048未测且无追加测速job |
| `q_prefill(l,b),q_decode(l,b)` | 正式估算公式所需的独立prefill/decode吞吐；当前只有短上下文（≤258token）4次greedy验证、合计≤256新token，没有专门长度/批大小测速job | 未安排独立测量；验证端到端生成时间不能拆作两项实测 |
| `t_EF(b)` | 当前float64的b=1/2/4：batch总秒数/b；无float32或更大batch当前任务 | TBD |
| `n_EF_relax[p50,p95]` | 当前FIRE/.05/40步有界弛豫的实际E/F数和是否收敛；正式BFGS/.03/300步分开 | 当前TBD；正式未测 |
| `t_relax[p50,p95]` | 当前有界弛豫秒数，含映射/去重主机开销、截断失败另表；不能当完整BFGS收敛时间 | 当前TBD；正式未测 |
| `t_IO,t_host` | checkpoint读写、数据准备/等待、canonical和结构映射开销；标持卡与非持卡时间 | TBD |

逐job预计值可按下式替换，取p50为中位情景、p95为保守情景，样本量不足时只作情景不宣称高分位可靠：

```text
H_LLM_est = [ΣL_LLM + T_seen/q_train + T_prefill/q_prefill
             + T_generated/q_decode + t_IO + 持卡等待秒数] / 3600
H_MLIP_est = [ΣL_MACE + N_sp*t_EF + Σ(t_relax_i)
              + 未含在t_relax中的持卡主机开销] / 3600
N_relax_feasible ≤ floor(min[剩余raw_EF/n_EF_relax_p95,
                            剩余MLIP秒数/t_relax_p95,
                            剩余动作/起点评估配额])
H_actual = Σ(allocated_GPUs × Slurm_ElapsedRaw / 3600)
W_two_cards ≥ max(H_actual/2, 最长依赖链墙钟)
```

混合长度按各桶分别求和；若端到端rollout吞吐已经包含prefill/decode，不再把二者重复相加。弛豫内部E/F时间已经在`t_relax`里，不能再乘一次`n_EF`加入；该数主要用于原始调用闸门和容量换算。主机预处理不持GPU时单列CPU成本和墙钟。预计完成不了计划样本时先缩样本或报告“无法判断”，不按吞吐倒推追加授权。[inference]

## 9. 逐轮代理偏差诊断与独立DFT终评

RSI每seed每阶段4个诊断末态，D0/R1/R2合计**最多36个316原子DFT单点名额**。最小四类探针的其余三个arm只在终点按每seed4个名额等额审计，再加最多36，故最小包诊断/审计合计**≤72名额**；纯搜索不伪造R轮。冻结/同数据的共享D0无需重复测。九arm扩展若对RSI和分轮搜索蒸馏逐阶段监控、其余七arm终点审计，则≤156名额；把所有九arm都按三阶段监控的极宽上界才是324，不作为默认申请。[inference]

共享D0与重合末态标签只实际计算一次，不虚增独立样本。终点审计从独立诊断流的最多4条8步轨迹中取样，在该arm参数冻结后执行；其MACE调用、动作/token/持卡成本已包含该arm限额，不能从top-k盲候选抽走同一个家族再两用。每次比较同协议相对能残差、低能尾部排序、DFT力与末态占位；报告训练前/后的残差分布，检查策略是否正在利用MACE偏差。诊断不能用来择优checkpoint、教模型DFT偏好或微调MACE；严重偏差时停止受影响的后继训练组并交审，已有失败与成本保留。[inference]

工程报警暂登记为：每轮4个样本中≥2个相对能残差超过0.6 eV/胞，或逐原子力误差95分位超过0.5 eV/Å，或新出现大量映射模糊/离格/同键多盆地。残差减去训练前固定参考偏移，不能每轮重拟合以消掉漂移；这些是筛查报警、不是统计误差保证。诊断额度未批准时不执行对应正式阶段，不能省略监控继续RSI。[inference]

盲终评协议继承审核07 G2/G3结果：G2四构型的弛豫相对能变化跨度<0.1 eV/胞且占位不变，才可用“MACE几何DFT单点+各method/seed等额top-k完整弛豫抽检”；否则全部top-k完整弛豫。G3三代表构型Gamma与2×2×2的成对相对能变化跨度≤0.05 eV/胞才放行Gamma；不达标统一加严。此条件选择在正式候选产生前锁定。[审核07；inference]

每method/seed给**k=5个不同、有效、未碰撞的末态规范家族**，按冻结MACE能量排序，平局按键字典序。每个家族只提一个预先规则确定的实际几何；不足5个不补免费搜索，缺项按失败计。最小四类×3seed最多**60**个入选名额；九arm扩展最多**135**。先做末态键/几何并集去重后统一DFT，实际计算数可能更少。DFT弛豫若又合并成同一末态，同method/seed仅最先登记名额计成功，其余仍留在分母，不补候选；跨method重合可共享标签，但不增加统计独立性。[inference]

若G2通过，初版完整弛豫抽检为每method/seed同等取2个名额：MACE排第1与第5（只有1个有效则只查1个，其余失败）；最小包最多**24**、扩展最多**54**个完整弛豫名额，并集去重后实际更少。任何抽检出现占位改变或相对能改动超已登记0.1 eV/胞协议范围，停止解盲结论，转为全部top-k完整弛豫或记“无法判断”；不能只给表现好的方法追加弛豫。若G2不通过，最小包最多60、扩展最多135个名额全部弛豫。[inference]

**主指标预登记：** `DFT_low_energy_rate = 在统一终评协议下 E_DFT ≤ E_min_reference + 0.30 eV/胞 的有效不同末态数 / 5`。`E_min_reference`来自训练前封存的共同参考池，成分/晶胞/精度/终评流程一致，评估端记录结构清单和阈值SHA；不是参赛方法事后最优能量。阈值数字由评估端固定，不用它指导训练。单点与完整弛豫若混用，主低能率用G2预定的统一单点协议；完整弛豫是等额稳健性检查，另表比较，不能只替换少数候选能量。[inference]

共同参考池的具体起步清单拟为既有16个大胞校准几何B00–B15，在正式训练前锁定并排除其训练家族；只由评估端持有标签。若G2/G3决定的最终协议与既有标签一致可复用，否则16个参考结构须统一补到最终协议，另计`H_reference_DFT≤16*h_reference`，该新费用未批。未取得一致参考标签时不解盲主低能率；不能临时改为各方法自己的最佳值。[inference]

DFT低能率是主比较，代理`δ`仅用于筛查和次指标；另报DFT最佳/中位相对能、达到预登记能档的资源、独特末态数、失败/碰撞率。最强搜索对手在验证集从SA/GA/CE选择并锁定；盲测同时展示全部arm，不在看过终评后删掉不利对照。[inference]

DFT成本另算，当前没有新池授权：

```text
H_diagnostic_DFT ≤ N_diagnostic_union * h_SP_diagnostic
H_blind_DFT(G2通过) ≤ N_union * h_SP_final + N_audit_union * h_relax_final
H_blind_DFT(G2不通过) ≤ N_union * h_relax_final
最小包：N_diagnostic_union ≤ 72；N_union ≤ 60；N_audit_union ≤ 24
九arm默认扩展：N_diagnostic_union ≤ 156；N_union ≤ 135；N_audit_union ≤ 54
全arm三阶段监控的极宽诊断上界324另列，不默认执行
```

`h_*`必须用G2/G3完成后的实际成本及另行登记的每job硬封顶替换；审核07的5卡时/构型是该次协议测试限额，不能当作已测普遍成本。DFT核时/GPU卡时、逐轮诊断、共同参考池、盲终评分别单列；不与60最小/162扩展训练探针或10冒烟互借，不把去重标签复用当成多份独立测量。[inference]

## 10. 三种结果与停止边界

预登记实用差异为低能率**20个百分点**（k=5时每seed一个名额）；它是研究决策阈值，不是材料能量误差。按同seed/起点配对，展示三seed的0–5成功数和差值；区间以独立运行/构型家族为单位，跨方法复用的同家族保留相关性。附报配对种子自助法95%描述性区间（重采样种子2026102399、10000次，只重采样完整seed组），并原样列出三个差值，不能把这种小样本区间当作稳健显著性检验。三seed的双侧精确符号检验最小p值仍为0.25；本探针的“支持”只指方向性支持，不作总体显著性结论。[inference]

| 结果 | 判据与允许主张 |
|---|---|
| 支持 | DFT主指标相对锁定最强搜索、冻结多采样、同数据一次SFT均有平均≥20个百分点的实用差异，三个seed的差值均非负且至少两个为正，抽检/诊断未推翻结论；只称本任务探针方向性支持分轮策略。完整研究仍需更多seed与预登记推断；没有L1不称泛化 |
| 不支持 | 有足够、收敛且一致的独立DFT证据，三个seed均未达到20个百分点实用收益，或强对手在至少两个seed反向达到实用差异；也包括MACE提升经充分DFT检验不能转化为收益。停止或收缩RSI算法主张，不以继续加轮次补故事 |
| 无法判断 | 预算先到导致有效数据/DFT不足、三seed方向分歧而不满足前述支持/不支持规则、数值误差/代理漂移过大、末态映射/盆地冲突、训练重叠未排除或DFT未收敛。完整研究的区间若跨实用阈值同样记无法判断；不把本机小胞或更低代理分数作为替代 |

所有方法的超时/非法/不足k进入对应分母；不能在盲测失败后换种子或新训练再选最好一次。每轮raw E/F、rollout token、seen-token、优化步、任一池/共同GPU限额先到就停；只有依赖失败项的后继作业停止，跨独立DFT/MLIP/训练池不连坐，超额或异常消耗才停整个受影响池。[inference；审核07]

## 11. 冒烟之后填写的交审栏

| 项目 | 实测/核验结果 | 当前状态 |
|---|---|---|
| 模型revision、tokenizer/template SHA、环境锁 | 逐文件记录见PREPARATION_STATUS、qwen_manifest；LLM/MACE版本见§4 | 远端下载/导入/tokenizer预检已通过；GPU运行待DFT后 |
| 0.6B可训练模块/梯度/显存/checkpoint | TBD | 待tinySFT |
| 训练/prefill/decode/加载吞吐 | TBD | 当前只测短样本tiny包；2048规模不在本包 |
| 316原子MACE单点/有界FIRE E-F分布、末态去重开销 | TBD | 待DFT后A100原版CUDA测速；完整BFGS仍未测 |
| 包装层严格JSON/末态键/计数/资源闸门 | 独立smoke包已有实现与静态预检记录 | 正式RSI状态机/审计整合仍待实现；GPU行为未测 |
| 按实测公式重算单RSI18/最小60/扩展162卡时内可完成数据量 | TBD | 替换§8参数后生成，不追加限额 |
| G2/G3终评协议与独立DFT额度 | TBD | 不阻塞计划撰写；阻塞正式相应终评 |

当前两job硬预留1.25卡时；其余8.75未分配，本文没有GPU测速数字。DFT之后回收两job实测及Slurm账单，只替换协议匹配的参数；正式60/162卡时方案和独立DFT池均继续未批准，成本估计保持暂定。

## 来源与查阅深度

- **[code]** [lattice.py](scripts/lattice.py)、[calibration.py](scripts/calibration.py)、[prepare_calibration.py](scripts/prepare_calibration.py)、[audit_calibration_geometry.py](scripts/audit_calibration_geometry.py)、[render_claude06_report.py](scripts/render_claude06_report.py)：已读动作、对称键、BFGS和全局末态映射源码。训练包装层不在现有代码中。
- **[code]** [maincell_timing.json](calibration/maincell_timing.json)、[requirements-lock.txt](requirements-lock.txt)、[models/manifest.json](models/manifest.json)：已有本机记录/锁。只支持Mac开发及权重来源，不支持A100吞吐。
- **[code]** 当前smoke包：[README](training_smoke/README.md)、[JOB_MANIFEST](training_smoke/JOB_MANIFEST.json)、[BUDGET](training_smoke/BUDGET.json)、[数据清单](training_smoke/data/manifest.json)、[sft_preflight](training_smoke/evidence/sft_preflight.json)、[PREPARATION_STATUS](training_smoke/evidence/PREPARATION_STATUS.json)：已读两job限额、分词统计、准备状态。16,023/258为CPU分词预检统计，不是GPU测速。
- **Q1 [direct_text]** [Qwen3-0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)：已读Model Overview、Quickstart、thinking开关和Best Practices，2026-10-23查阅。
- **Q2 [direct_text]** [Qwen3-1.7B 官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)：已读参数规模/层数、兼容提示和非thinking接口，2026-10-23查阅；未下载/训练权重。
- **Q3 [code]** [官方 tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)：已读chat_template中的空thinking前缀条件；正式运行另锁revision/SHA。
- **Q4 [direct_text]** [PEFT官方LoRA文档](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora)：已读`LoraConfig`的r、target_modules、alpha/dropout/bias语义及冻结基座说明。只支持API/方法定义，rank16等设置属本项目设计。
- **Q5 [direct_text]** [Qwen官方LLaMA-Factory页](https://qwen.readthedocs.io/en/latest/training/llama_factory.html)：明确提示待更新Qwen3、当前示例为Qwen2.5；未作为Qwen3已验证训练教程使用。
- **[code/official_index]** 当前0.6B已从[官方ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)下载并核对，逐文件证据见[qwen_manifest](training_smoke/evidence/qwen_manifest.json)；[1.7B官方入口](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B)仅为后续索引，当前未安排运行。
- 其余预算、采集/对照设计、统计阈值、排程与选择依据均为 **[inference]**，不冒充官方建议或本任务实测结论。
