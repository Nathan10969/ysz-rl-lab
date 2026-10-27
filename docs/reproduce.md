# 公开包的复现范围

## 第一层：检查仓库和保存结果

Python 3.10+，使用根目录 `requirements-cpu.txt`。运行 README 中的检查、测试和结果复算命令即可。

- `tools/check_publication.py`：检查公开文件范围、凭据形态、源码语法、来源哈希和整理文档链接。
- `tools/results.py`：用保存的逐起点能量重算均值、起点重采样标准误差及配对差。
- `examples/inspect_state.py`：用真实晶格代码生成组成，列出合法动作并打印局部特征。没有能量模型调用。
- `tests/`：验证导出、配对起点校验和公开结果一致性。

## 第二层：研究代码

`research/` 保留源项目中的相对目录与模块名。`formal12/scripts/` 是当前正式实验代码快照；`scripts/` 包含早期版本、预检和运维脚本；`tests/` 是已有研究测试。

部分旧测试需要完整的本机证据包或训练产物，本公开包没有把这些测试全部作为可运行承诺。可以先运行不依赖模型的已有测试：

```bash
PYTHONPATH=research/scripts python -m unittest discover -s research/scripts -p 'test_lattice.py' -v
PYTHONPATH=research/scripts python -m unittest discover -s research/scripts -p 'test_calibration.py' -v
```

完整训练还需要 PyTorch、Transformers、PEFT、safetensors、MACE 及模型资产。环境原记录见 [environment-original.md](experiments/environment-original.md)，物理环境锁文件见 [requirements-lock.txt](../research/requirements-lock.txt)。锁文件是当时环境的记录，不是所有平台的通用安装承诺。

源代码中的 `/LOCAL_HOME`、`/CLUSTER_HOME`、`your-cluster` 是脱敏替代值。使用者应为自己的环境重新配置路径、哈希清单和调度脚本。不要直接运行历史提交脚本。

## 第三层：物理实验

主任务依赖冻结的 MACE 评分器与 Qwen3-0.6B。DFT 依赖单独准备的 VASP 环境和赝势。模型权重、赝势、训练 adapter、第三方论文 PDF 和集群环境没有包含在这个仓库中；获取与许可入口见 [第三方说明](../THIRD_PARTY.md)。

逐起点结果保留初始能量和最佳能量，支持评价指标复算；公开摘要没有包含全部中间坐标、逐次力或原始前向日志。因此它不等价于从头重跑物理实验，也不能补回原日志未保存的信息。

## 数值口径

- 主体系指标：MACE eV／316 原子超胞。
- 每个起点取 4 条轨迹的最好下降量，再对起点求平均。
- 标准误差：对起点重采样；配对比较先对齐起点编号和初始能量。
- 复算脚本沿用 NumPy `default_rng(0)` 的 2,000 次均值重采样；配对用 `default_rng(1)` 的 4,000 次重采样。
- 配对计算使用未舍入数值。原研究 `val_metric.py` 曾先把逐起点值舍入至 0.001 eV；公开工具保留精度，可能带来末位差异。
- 训练 rollout 分数与独立验证分数用途不同；不能把所有结果文件混成一张“测试性能”表。
