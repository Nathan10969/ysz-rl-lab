# 首版公开包验证

验证日期：2026-10-26，America/Los_Angeles。使用新建的独立 Python 3.12 虚拟环境，安装根目录 `requirements-cpu.txt`，不复用研究目录的训练环境。

| 检查 | 结果 | 范围 |
|---|---|---|
| `python tools/check_publication.py` | PASS | 公开文件、凭据形态、本机标识、整理文档链接、JSON、Python 语法和来源哈希 |
| `python -m unittest discover -s tests -v` | 9 项通过 | best-of-four 指标、起点对齐、失败数据拒绝、来源完整性、时间戳空更新、导出遇凭据时保留原文件 |
| `PYTHONPATH=research/scripts python -m unittest discover -s research/scripts -p 'test_lattice.py' -v` | 3 项通过 | 电中性、组成、对称 key、动作组成守恒 |
| `PYTHONPATH=research/scripts python -m unittest discover -s research/scripts -p 'test_calibration.py' -v` | 3 项通过 | 大胞组成、平移 key、历史校准任务设计 |
| `python examples/inspect_state.py` | PASS | 324 母位点、316 原子、390 个合法动作；物理模型调用为 0 |
| 再次运行导出且输入不变 | 保持不变 | 不修改时间戳，不制造空更新 |
| 远端仓库与本地主分支 | 已核对一致 | GitHub 可见性为 PUBLIC，默认分支 main |

源码快照 193 个白名单文件，保存结果摘要 59 份，均提供源哈希和公开副本哈希。公开复制的研究源码经过路径／账户脱敏；这一层是可阅读和追溯的快照，完整生产训练仍需要原模型、证据包与集群配置。

这些检查验证公开包的完整性与可运行的 CPU 部分，不把它们记作新增物理实验、模型训练通过或最终 DFT 结论。
