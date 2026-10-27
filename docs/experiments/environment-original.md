# 训练环境规格：审核08末态状态、局部动作与GRPO

现行补充：审核10已获用户批准，在原10卡时冒烟池内新增总封顶6卡时，用于小规模v2教师采集/正式BFGS测速、格式增强、v2 SFT及80例检查、通过后一次GRPO。旧648成功、650因未补v2 SFT而0/20退出，均保留原费用和证据。新任务必须先通过与作业同路径的真实CPU batch1/4单点和短BFGS预检；阶段失败即停，不重试。正式训练/新DFT未批。当前登记见CLAUDE10_EXECUTION.md。

日期：2026-10-24（America/Los_Angeles）。协议仍为 `YSZ08_terminal_local2_v1`，审核09通过设计框架但未批准正式额度。原633完成；635接口失败后按明确授权仅替代一次648，MACE batch1/4/16通过；唯一GRPO650真实GPU生成0/20，未进入MACE/PG并停止。原51项CPU验收为历史，补充59项回归和真实CPU预检见training_smoke/evidence/review09_code_verification.json；原环境未修改。当前状态见CLAUDE09_RESULTS.md；训练扩大目标与未测边界以TRAINING_PROBE_PLAN.md为准。旧审核07版本仅作历史。

## 1. 状态和转移

主任务固定15.42 Å立方胞、324母格点，物种Zr92Y16O208，316真实原子、8氧空位。`reference(3)`编号0–323，不假定前108个全是阳离子。Zr/Y/O/空位用40/39/8/0，送物理模型前移除0。

完整环境状态保存当前末态占位z、实际弛豫几何R、E与历史最低B、剩余8步/弛豫/E-F/token/时间预算、最近3步动作与能差、组开始时冻结的历史末态集合V和回合内访问集合。LLM只见紧凑JSON摘要，日志保存完整状态、几何hash与协议SHA。新颖性依赖访问集合，不能只保存原子编号就声称状态完整。

```text
当前弛豫末态占位 z_t
  → 合法局部动作 → 待评估占位请求 c
  → 确定性母晶格重建 → 冻结MACE弛豫 → R_(t+1), E_(t+1)
  → 周期性一对一投影 → 新末态 z_(t+1), canonical key
  → 下一动作在 z_(t+1) 上执行
```

v2不继承上一步连续几何，而是从动作后的占位c重建理想坐标再弛豫；这是一致的构型优化环境，不是扩散动力学。所有臂必须同规则；不与旧版“持续在原初态c上动作”的成绩直接连线。

## 2. 动作与合法率

严格JSON字段仅 `op,sites`，op只能 `swap_cations` 或 `move_vacancy`；sites为两个不同JSON整数，拒绝bool/float/越界/重复字段/额外文本。交换元素分别恰为Y/Zr或空位/O，必须同母子晶格，周期性距离≤该子晶格第二壳，容差1e-6 Å。

| 子晶格 | 第一壳/第二壳距离，Å | 允许动作 |
|---|---|---|
| 阳离子 | 3.63452886 / 5.14000000 | Y与Zr交换 |
| 阴离子 | 2.57000000 / 3.63452886 | 空位与O交换 |

距离直接来自固定无缺陷母晶格最小像，不从弛豫缺陷几何估计。无Y–空位NN禁令，不把材料偏好写入合法性。非法消耗一步/生成token/持卡时间，−1，不调MACE、不修复重试；K=8包含这类尝试。

正式SFT闸门为独立v2状态76/80合法；GRPO冒烟闸门19/20，后者在该≤2卡时作业内先检查。只测动作合法性，无物理查询。633的四次格式生成不证明v2局部动作合法率；失败不自动补训练。分母和样本量如实报告，不称总体合法率的置信下界。

## 3. 语言模型与概率口径

Qwen3-0.6B，LoRA r16/alpha32/dropout.05、q_proj/v_proj，BF16起步，基座冻结。模型权重SHA `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`，tokenizer_config SHA `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`。实际revision/其余文件指纹见 training_smoke/evidence/qwen_manifest.json。

远端已准备torch2.7.1+cu128、transformers4.57.1、peft0.17.1、accelerate1.10.1；MACE独立overlay为mace0.3.16、ASE3.29.0、spglib2.7.0；这些只是准备记录，GPU行为待测。不修改已有共享环境。

官方chat template，`enable_thinking=False`。新动作≤64 token，prompt≤1984，总上下文≤2048；必需占位列表不截断，超长报编码失败。v2正式学习臂/GRPO冒烟采用temperature=1、top_p=1、top_k=0完整softmax，采样/old/current/reference概率一致；采样与概率评估统一eval模式以关闭dropout，eval并不关闭梯度。固定随机流，不能用约束解码掩盖合法率。

动作内容及EOS是唯一策略target；system/user/环境结果/历史观察/padding不参与PG或KL。生成前缀中的空think块属于prompt，不误判成新输出。保留不可改写的old log-prob与SFT参考适配器；关闭LoRA得到的基座不是SFT参考。

## 4. MACE、弛豫与末态

冻结原版CUDA MACE-MP-0b2 small float64，SHA `d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9`。固定胞BFGS、maxstep=.1 Å、fmax=.03 eV/Å、最多300步。模型加载、构图、传输、每次真实前向和失败均计成本。

可训练物理结果须能量/力有限、力收敛、组成与末态匹配有效；截断/未收敛不能用最后一帧冒充成功。阳离子Y/Zr联合匹配108位，氧匹配216位，周期性最小像全局一对一最短指派；未分配阴离子位是空位，不能按原始site_id宣称末态。母晶格5184个对称排列下取canonical key；保留assigned_site_ids、距离、geometry hash、跨位及NN/NNN。

初版最大匹配距离小于对应子晶格最近邻距的一半，代价差≤1e-8 Å²的不同占位平局标模糊；阈值在正式采集前冻结，不能为低能结果放宽。DFT使用实际末态R和统一原子顺序，不使用映回的理想坐标。

635最新用户修订为SP/FIRE batch1/4/16，8输入复用、21个弛豫副本，FIRE/.05/40、raw E/F≤1024和原.75卡时不变；这些受限测量不等于正式BFGS/.03/300成本，也不换掉已提交脚本。

## 5. 请求缓存和终态去重

请求键包含成分/晶胞、实际编号的动作后占位c、模型SHA、dtype、初始几何和弛豫参数。已知且核验的request→terminal记录才可直接命中；未知请求必须弛豫，随后映射和去重。同terminal key不同几何盆地不合并成同一个缓存结果；能量跨度>.01 eV/胞或结构冲突入诊断。

缓存按arm/seed隔离，仅共同D0明确共享并归因。对称缓存须有可还原的正逆变换；尚未实现时仅复用相同编号请求。不同初态落入同末态只增加已花费用，不增加候选名额。跨方法同几何DFT可共享物理账，不能回填另一方法的发现或提高统计独立性。

## 6. 奖励、计数、预算

能量项为历史最佳降幅/.20 eV，新末态+.05，非法−1；合法物理失败单列−1且保持状态。只有能量项总和等于全回合最佳降幅/.20，总回报还含新颖性/失败。组开始前冻结arm/seed历史V，四回合各自拷贝，完整组结束才合并新key；循环访问不给重复奖励。回合级G4/K8，主优势为组内总回报标准化，clip.2、KL到固定SFT系数.03，仅动作token。零方差不重采样造数据，记录未更新。

主物理计数 `N_relax_actual` 在真实启动弛豫时加1：失败/未收敛照计，batch逐结构计，cache hit计0，事后重复不退款。另记N_attempt/N_cache/N_unique与raw E/F；E/F按每结构每真实前向计，batch b计b，缓存返回0，线搜索/预热/失败均计。实际卡时按父Slurm AllocTRES×ElapsedRaw；CPUTimeRAW单独核时。任一硬闸门先到就停，不能靠更多免费命中无限推理。

## 7. 验收合同与交审

本机代码已完成：claude08_env.py、claude08_data.py、claude08_grpo.py、smoke_grpo08.py及对应测试。51项新代码测试正常与优化模式通过，规格与最终质量审核均通过；现有回收流程另做105项回归。源码SHA和实际命令见 training_smoke/evidence/claude08_cpu_verification.json。CPU测试覆盖以下合同，真实GPU上的梯度、吞吐、显存和保存重载仍待实测：严格JSON与两壳动作、组成守恒；动作作用于末态；请求命中与事后重复不同计数；同组样本执行顺序不改新颖性；能量项望远镜和循环0收益；非法−1无MACE；zero-std有限；PG正负优势符号；old/reference冻结；观察token梯度mask；有限梯度/参数变化/保存重载；raw E/F与持卡时间边界；家族隔离。

唯一≤2卡时GRPO650已在633与648证据及真实CPU预检通过后提交，保持1seed、4起点、G4/K8、最多1次更新；实际0/20，未更新并停止。没有第二次GRPO授权。项目最多1个GPU作业，LLM/MACE/DFT共用这个单卡上限。全部既有11项不可重提；无正式RSI/baseline、1.7B、L1或新DFT授权。日志与产物交本地人工审核，不自动联系外部代理。

## 一手来源

[审核08](REVIEW_CLAUDE_08_RL_arm.md)给出设计；[DeepSeekMath](https://arxiv.org/html/2402.03300v3)支持GRPO组内基线、无critic与参考KL的基本机制；具体材料状态、缓存与物理预算为本项目设计。[ReST](https://arxiv.org/abs/2308.08998)支持自采样加筛选/离线策略改进的谱系解释。

## 8. GRPO共同运行环境的准备闸门

只读核对发现：现有LLM解释器有torch/transformers/peft但缺MACE/ASE/spglib/SciPy，现有MACE解释器缺transformers/peft；两者不能直接冒充完整GRPO环境。原633/635环境不改。两项回收及证据闸门通过后，再为唯一GRPO准备独立依赖overlay，核对共同torch/CUDA与以上模块import、模型/代码SHA；依赖准备不加载GPU模型或评分。代码CPU测试不能代替这个实际环境预检。

当前运行事实：635测速在第一次E/F前因图数据接口不兼容失败，GPU启动9秒照计0.0025卡时。其batch等价性未验证，故GRPO前置闸门不通过；当前不准备共同依赖overlay、输入或新作业，不自动重试。上文CPU验收仅指环境/GRPO源码合同，不能代替失败的真实测速。
