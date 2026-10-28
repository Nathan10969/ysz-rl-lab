# 外部软件、模型和论文

项目使用以下上游工作。它们不属于本项目原创成果，且具有各自的许可和使用条件。

| 上游 | 本项目中的用途 | 官方入口 |
|---|---|---|
| MACE | 冻结的原子间势、结构弛豫、能量和力 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| Qwen3-0.6B | 语言模型基座 | [官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B) |
| ASE | 原子结构与几何工具 | [ASE](https://ase-lib.org/) |
| spglib | 晶体对称性 | [spglib](https://spglib.github.io/spglib/) |
| PyTorch、Transformers、PEFT | 张量、模型加载、LoRA 训练 | [PyTorch](https://pytorch.org/) · [Transformers](https://github.com/huggingface/transformers) · [PEFT](https://github.com/huggingface/peft) |
| VASP | DFT 校准与验证 | [VASP](https://www.vasp.at/) |
| DeepSeekMath / GRPO | 组内相对策略优化的方法来源 | [原论文](https://arxiv.org/abs/2402.03300) |
| Dong、Qi、Li、Chen，Acta Materialia 127 (2017), 73–84 | 相关前作：用局部配位偏好引导 8YSZ 低能构型搜索，帮助界定本项目贡献；未打包论文全文 | [原论文](https://arxiv.org/abs/1701.05089) |

公开仓库包含本项目的调用、控制、测试与分析代码，不包含外部模型权重、训练 adapter、VASP 二进制／赝势、论文 PDF 或整份第三方源码。源工作区中的 MACE 上游补丁及缓存也没有直接打包。

Qwen 模型卡标注 Apache-2.0；具体使用以下载时的官方模型卡和许可文件为准。代码包、模型资产和数据的许可需要分别阅读，本仓库的 MIT 许可不替代上游许可。

历史研究文档保留当时引用和技术判断；公开副本已替换本机用户路径、集群账户与主机信息，源哈希与公开文件哈希见 `provenance/manifest.json`。
