# 训练环境规格：YSZ 主胞、冻结 MACE 与末态记账

日期：2026-10-23。状态：**正式RSI环境为可审阅设计，完整包装层尚未接好；独立两job冒烟包已准备，GPU测量待DFT前驱结束后运行。本文不提交或运行作业。**

依据：[REVIEW_CLAUDE_07_gates.md](REVIEW_CLAUDE_07_gates.md)、[PLAN.md](PLAN.md)、[lattice.py](scripts/lattice.py)、[calibration.py](scripts/calibration.py)。实验与额度见 [TRAINING_PROBE_PLAN.md](TRAINING_PROBE_PLAN.md)。来源标记：`code` 是已读源码或已有文件，`direct_text` 是官方正文明确陈述，`official_index` 是官方目录，`inference` 是本规格的设计，尚非实测结果。

## 1. 用途与必须分开的两个状态

智能体的工作是改变 Y 和氧空位的排布，寻找冻结 MACE 势认为较低能的结构；DFT 是另一个独立终评器。MACE 是机器学习原子间势（MLIP），根据原子坐标计算能量和力；弛豫是沿力调整坐标，直到原子基本不再受力。

**动作作用于初始占位，评分、材料统计、候选选择、训练筛选与去重全部作用于弛豫末态。** 两个不同起点可能弛豫到同一个末态，只算一个候选。[审核07；code/inference]

设 `c_t` 为第 t 步供动作操作的格点占位，`R(c_t)` 为统一几何起点经冻结 MACE 弛豫得到的坐标，`z_t=map(R(c_t))` 为末态占位，`K_t=canonical_key(z_t,3)` 为末态规范键。环境同时保存这些对象，不用一个 `state` 字段混装。下一动作交换 `c_t` 的两个格点，得到 `c_(t+1)`；不会把 `R(c_t)` 的坐标隐式带进下一步，也不会暗中用 `z_t` 替换 `c_t`。[inference]

```text
初态占位 c_t → 合法动作 a_t → 新初态占位 c_(t+1)
                                    ↓ 标准母晶格重建
                             冻结 MACE 固定胞弛豫
                                    ↓
                    末态坐标 R、末态占位 z、末态键 K
                                    ↓
                         末态能量、去重、回报与日志
```

## 2. 组成、编号与动作合同

| 项目 | 固定规格 | 来源 |
|---|---|---|
| 主任务 | 3×3×3 常规萤石胞，108 阳离子位、216 阴离子位，324 格点；Zr92Y16O208，316 个真实原子、8 空位 | PLAN；code |
| 晶胞 | 常规胞 a=5.14 Å，主胞边长15.42 Å，固定胞；所有方法相同 | `calibration.reference(3)`；code |
| 编号 | 0–323，来自 `reference(3)` 的实际格点表；不能假定前108个编号都是阳离子 | code |
| 占位 | Zr=40、Y=39、O=8、空位=0；`X`/0 只在占位表中存在，送入 MACE/DFT 前移除 | code |
| 电荷补偿 | 16 Y、8 空位恒定；不加 Y–空位 NN 排除，不预写 NNN 或方向偏好 | PLAN；inference |
| 94原子场 | 仅开发/诊断；不替代316原子主任务结果 | PLAN |

每次只接受一个严格 JSON 对象，没有解释、代码围栏或额外字段：

```json
{"op":"swap_cations","sites":[3,24]}
```

表示交换一个 Y 与一个 Zr。示例编号仅说明格式，是否合法由当前 `c_t` 和实际格点表决定。

```json
{"op":"move_vacancy","sites":[41,77]}
```

表示交换一个空位与一个 O。允许同子晶格内任意远近的两个位置；这是构型优化操作，不能解释成真实扩散时间或迁移能垒。[inference]

包装层必须检查：字段恰为 `op,sites`；`op` 恰为上述两值；`sites` 恰有两个 JSON 整数（拒绝布尔值、浮点、字符串）；两者不同且在0–323；当前元素对恰为 `{39,40}` 或 `{0,8}`；物种必须符合母晶格子晶格。输入超过64个新生成 token、截断、重复字段、NaN、额外文本均为非法。**`lattice.apply_action` 已有元素对/范围/同位校验，但不是完整严格 JSON 解析器，且 `bool` 在 Python 中属于 `int`；不能直接把该函数当成全部合同已实现。**[code/inference]

非法动作不改变 `c_t`，不调用 MACE；仍消耗一次尝试、生成 token、推理时间和轨迹一步。错误仅返回标准错误码，不免费修复、不偷偷重采样。每轨迹至多8次动作尝试；失败与重复也占配额。[inference]

## 3. 观察与语言模板

观察使用紧凑 JSON，至少包括：`schema_version`、模板/晶胞ID、当前初态 Y/空位编号、当前末态 Y/空位编号、二者是否不同、当前末态与历史最佳末态的相对 MACE 能量、最近4步动作/末态能差、剩余尝试/E-F/token/时间预算、末态配位统计。格点坐标由固定模板给出；可按确定规则附相关局部邻域，不展开整个 CIF。当前初态列表用于下一动作，末态列表用于解释能量与邻近关系。[inference]

`calibration.shell_counts` **当前只实现 Y–空位 NN/NNN 两壳**。第三壳、Y–Y/空位–空位、局部邻域和编号变换是待实现项；若加入，所有基线同时可见且计算时间记账。不能把初态壳统计作为末态壳统计。禁止观察包含 DFT 标签、未来动作的免费能量预览或盲终评阈值的数值标签；任何候选能量查询均进入 E/F 实账。[code/inference]

Qwen 输入使用下载版本自带 `tokenizer.apply_chat_template`，推理时 `add_generation_prompt=True, enable_thinking=False`。官方模板在关闭 thinking 时可能在**输入前缀**放置空 `<think>` 块；解析器只校验新生成的动作文本，不把模板前缀误判成非法 JSON。训练时使用同版本模板构造一致的 assistant 前缀，只对动作内容及结束符计算损失，prompt/固定前缀全部 mask；用具体 token ID 测试边界，不能靠字符串长度切分。[direct_text/code；Q1/Q3；inference]

输入至多1984 token，新输出至多64 token，总长不超过2048。超长输入按固定规则先缩历史和邻域，不截断必需的 Y/空位列表；仍超长则报编码失败并记账，不改序列上限。采样固定为非thinking的 `temperature=.7, top_p=.8, top_k=20`，随机数由运行种子控制；不把该设置称为任务最优值。[direct_text；Q1；inference]

本节是正式环境拟议上限；当前获准tinySFT包实际序列上限512、数据最长258token，32优化步/64微批、batch1/累积2，仅4条greedy验证生成、合计≤256新token。没有2048token扩展或独立prefill/decode测速job，不把当前短样本吞吐说成已测正式序列吞吐。[code；smoke包预检]

## 4. 固定 MACE 弛豫与有效标签

1. 每个 `c` 从 `reference(3)` 的同一理想格点重建，移除0空位，保留 `site_id` 和物种；几何不继承搜索历史。探针初版无随机微扰；若以后采用按初态键确定的微扰，必须换协议ID并给所有方法相同规则。[inference]
2. 冻结 MACE-MP-0b2 small，SHA256=`d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9`。A100 使用原版 CUDA MACE，主评分 dtype先定float64，与校准几何协议衔接；Mac 的 MPS 补丁只用于本机开发。float32只能在吞吐测试中单列，未经能量/末态一致性检查不得替换主评分器。[PLAN、manifest；code/inference]
3. ASE BFGS，固定晶胞，`maxstep=.1 Å, fmax=.03 eV/Å, steps≤300`；对应现有 `prepare_calibration.relax`。初态能量、每次实际 E/F 评估、末态能量/力和耗时全部记录；300离子步不等于300次E/F。[code/inference]
4. 仅在弛豫收敛、能量/力有限、组成正确、映射有效时输出可训练的末态标签。超时、达到原始调用限额、未收敛和异常力分别记录，不能把最后一帧当作成功候选。失败原始日志保留，已花成本不退回。[inference]
5. 盲DFT使用实际 MACE 末态几何及统一原子顺序；不把映回格点的理想坐标送作“相同几何”终评。[inference]

以上BFGS/.03/300步是正式评分协议，仍未获正式训练额度。**当前smoke测量采用FIRE/.05/每胞40步**：316原子、float64，单点batch1/2/4，6个有界弛豫胞，总raw E/F≤1024；达到步数上限未收敛标`step_capped_unconverged`，不作为收敛标签。smoke末态仍映射并按末态canonical key去重，但测速不代表正式BFGS完整收敛成本。[code；smoke包]

模型初始化、预热、dtype比较以及失败弛豫中的物理评估都计入相应池。生产缓存命中可以为零次新E/F，但推理、哈希、映射及等待GPU占用时间仍记账。[inference]

## 5. 末态映射、规范键与缓存

沿用 `audit_calibration_geometry.py` / `render_claude06_report.occupancy` 的思路：周期性最小像距离下，阳离子联合匹配108个阳离子位（不把 Y 和 Zr 分开限制），O匹配216个阴离子位，分别做全局最短指派；未分配的8个阴离子位为空位。不得用原始 `site_id` 直接宣布末态占位，因为原子可能跨位。[code/inference]

在无缺陷母晶格上求对称排列，调用 `calibration.canonical_key(mapped,3)`；主胞5,184个格点排列来自既有代码/检查。规范键等同于“把旋转、平移等价的排布统一成同一编号”。初态键只用于重建和缓存索引，**不能用作结果去重键**。[code/inference]

| 键/对象 | 用途 | 禁止用途 |
|---|---|---|
| `initial_request_key` | 成分/晶胞/初态规范键/实际编号占位哈希/几何协议/模型SHA/弛豫参数/dtype的请求缓存；可快速复用已完成弛豫 | 独特终态计数、成功率分母 |
| `terminal_key` | 弛豫后映回占位的规范键；候选家族、回报、训练筛选、配位统计、跨方法DFT去重 | 忽略同键下的几何冲突 |
| `geometry_hash` | 实际末态几何与原子映射的审计、DFT输入复现 | 替代规范键把同占位家族扩成多个“独立成功” |

每个 `terminal_key` 建一条结果记录，关联所有初态请求及实际末态几何。相同末态键不增加新颖性、训练家族权重或DFT配额。保留该方法**已经实际得到的**最低有效末态能量作为其家族成绩；不把另一方法后来找到的更低几何回填为该方法的发现。全局DFT并集可复用完全相同几何的标签，但不回填发现时间。[inference]

初版请求缓存只复用**相同编号占位**，故包含实际编号占位哈希；对称等价但编号不同的请求仍可重新弛豫并计费，结果再按末态键合并。以后如加对称缓存，必须保存正/逆空间变换并将坐标和末态编号还原到当前观察坐标系，不能只复用能量而错配动作位点。搜索用MLIP缓存按method/seed隔离，仅共同D0可以明确共享并归因采集成本；盲DFT并集去重不等于把其他方法的搜索标签免费送入当前策略。[inference]

同键几何若能量跨度>0.01 eV/胞，或结构匹配显示不同局部盆地，标记 `terminal_family_ambiguous`，先进入诊断队列；不靠选最好数值静默消解，也不在SFT中重复放大。离格结构或几何匹配失败时保留原始几何和成本，报告比例，暂不进入占位任务的有效训练/终评分子。[inference]

映射质量的初版保守闸门：每原子分配距离须小于对应母子晶格最短位间距的一半；比较次优指派，存在占位不同而总代价差≤1e-8 Å²的平局则标记模糊。闸门是工程筛查，非材料定理；若拒绝率偏高，应单列离格任务，不能通过改阈值挑选更低能结果。容差与结构匹配参数须在正式采集前冻结。[inference]

## 6. 回报、轨迹选择和停止

实际评分为有效末态能量 `E_MACE(R(c))`，同成分/同协议减去固定参考。训练轨迹回报取 `E_start_terminal − min(E_terminal_seen)`；末态最佳记录仅每个规范家族一次。允许中间升能步；以完整轨迹的末态最佳回报筛选，不能仅保留逐步降能动作。[inference]

同一末态重复访问的新增收益为0，不重复发新颖性奖励。非法/未收敛为标准失败码，SFT不把失败动作当正确答案；评估时仍占尝试与失败分母。最多8步，或阶段E/F、生成token、seen-token、优化步、GPU时间任一硬封顶先到即停。不能在限额耗尽后再免费完成当前弛豫；启动前预留收尾/检查点时间，不完整样本只记成本。[inference]

代理改善解释采用 `δ_proxy=max(0.30 eV/胞,3×数值相对能误差)`。这是校准经验支持的**实用筛查门槛**，不是单个结构误差上界、置信区间或对新分布的保证。小于门槛可以记曲线，不能据此宣布真实收益；主比较由等额 top-k 独立DFT低能率决定。[审核07；inference]

## 7. 数据隔离与最小日志

训练、验证、逐轮DFT诊断和盲终评使用不同起点清单/种子流；按末态规范键与共享轨迹形成的连通家族隔离，坐标/编号增强始终跟随原家族。若后来发现跨集合碰撞，整条相关轨迹从新SFT候选中隔离；若已训练，盲评端将该家族列为碰撞失败，不可通过重命名换编号保留为未见结果。校准DFT、逐轮诊断DFT和盲DFT标签均不进入代理赛道D0/R1/R2的prompt、示范筛选、CE拟合或SFT。[inference]

每次尝试至少记录：运行/方法/种子/轮次/轨迹/步编号，模型与模板SHA、初态键、末态键、几何哈希、动作原文/解析结果、输入/输出token、原始E/F累计数、弛豫步数/收敛/fmax、初/末能量、跨位/映射质量/重复标志、实际CUDA时间与端到端时间、缓存命中、检查点与数据清单SHA、失败原因。每次批量调用记录批大小及实际结构数。[inference]

原始 E/F 定义为“一份316原子结构在一次新的几何上被势模型实际评估一次能量/力”。同一前向同时给E和F只记1；分两次前向记2；batch中b个结构记b，不能记1。ASE缓存返回不算新前向，预览/失败/线搜索/复核算。独立smoke包已按实际前向/batch实现计数与1024闸门并做静态预检；正式RSI包装层仍需接入同口径，不能由此声称GPU计数行为已实测。[code/inference]

## 8. 待实现项与验收

可复用：母晶格/组成、对称键、移除空位、基本合法换位、已有全局末态映射、固定BFGS协议。[code]

正式RSI待实现/整合：严格JSON包装层；`c/R/z/K`分离的状态机；末态家族缓存；原始E/F计数及资源闸门；家族隔离清单；Qwen模板/损失mask；不可改写的轨迹账本。独立smoke已有动作样本校验、模板/mask、末态映射、计数和限时实现及CPU/静态预检，未成为正式RSI环境；GPU行为与吞吐仍待测。[code/inference]

本机验收覆盖：两个合法动作组成守恒；跨子晶格/同位/bool/额外字段拒绝；交换前后初态变但跨位后相同末态只计一次；对称变换末态同键；同初态不同历史重建相同；原始batch计数与失败计费；限额边界不超支；训练与盲评家族碰撞处置；模板生成前缀、动作target及padding全部正确mask。验证包装层不需要使用保留DFT标签。[inference]

## 9. 当前获准冒烟的执行边界

当前只有两个单卡job：`c07_smoke_sft`硬封顶0.50卡时（30分钟），`c07_smoke_mace`硬封顶0.75卡时（45分钟）；各6CPU、90GiB主存，合计**1.25卡时预留**。独立10卡时池的**8.75未分配，不自动复测、扩批或运行1.7B**。SFT为64训练+4验证，计划训练输入16,023非padding token、单条最长258、序列上限512、seen-token硬限32768；该数是tokenizer预检统计，不是实际已训练量。MACE范围见§4，不含float32、batch8/16或正式BFGS评分数据采集。[code]

Qwen逐文件revision、tokenizer/chat-template SHA及模型权重已在CPU/文件预检阶段核对，见 [qwen_manifest.json](training_smoke/evidence/qwen_manifest.json) 和 [sft_preflight.json](training_smoke/evidence/sft_preflight.json)。这与GPU模型载入/前向/训练是否成功是两项状态；后者仍待DFT之后测量。[code]

两job排在已提交的GPU DFT八项617/619/621/623/625/627/629/631终态之后；CPU补跑1890508另属CPU池。冒烟ID/队列状态以父任务提交账为准，当前没有GPU冒烟测量结果；不调用本文变更提交、取消或重试任务。正式60/162卡时方案仍未批准，短序列/有界FIRE结果不能自动放行正式实验。[code/inference]

## 来源

- **[code]** 本地 [lattice.py](scripts/lattice.py)、[calibration.py](scripts/calibration.py)、[prepare_calibration.py](scripts/prepare_calibration.py)、[audit_calibration_geometry.py](scripts/audit_calibration_geometry.py)、[render_claude06_report.py](scripts/render_claude06_report.py)。已读源码；不是本文新实现。
- **[code]** 当前两job的[smoke README](training_smoke/README.md)、[JOB_MANIFEST](training_smoke/JOB_MANIFEST.json)、[BUDGET](training_smoke/BUDGET.json)、[sft_preflight](training_smoke/evidence/sft_preflight.json)、[PREPARATION_STATUS](training_smoke/evidence/PREPARATION_STATUS.json)，以及[smoke_mace.py](scripts/smoke_mace.py)、[smoke_sft.py](scripts/smoke_sft.py)中的计数/步数配置：这里只核对准备文件与源码，未执行GPU计算。
- **Q1 [direct_text]** [Qwen3-0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)，2026-10-23查阅：关闭thinking开关、非thinking采样建议、Transformers兼容提示。仅支持接口事实，不支持本任务准确率/吞吐。
- **Q3 [code]** [Qwen3-0.6B 官方 tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)，已读`chat_template`：关闭thinking的generation前缀含空think块。正式运行另冻结下载revision与文件SHA；网页main不是永久锁。
- 其余规范与数值阈值均为 **[inference]**，应随正式实现清单一并审阅。
