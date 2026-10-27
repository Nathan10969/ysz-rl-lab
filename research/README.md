# 研究代码快照

- `formal12/scripts/`：当前正式实验的几何、环境、模型 actor、SFT、GRPO、搜索、菜单与约束解码代码。
- `scripts/`：早期实验、校准、预检、测试与运维代码。
- `tests/`：已有研究测试；其中一部分依赖完整私有工作区和训练产物。
- `formal12/val_metric.py`：原研究评价脚本，保存原实现以便追溯。
- `formal12/greedy_weights.json`：D0 上拟合后冻结的特征贪心系数。

源文件在公开副本中只做用户路径、集群标识等脱敏替换，原始与公开文件哈希见 `../provenance/manifest.json`。
原生产环境、历史提交控制及未随仓库提供的模型／数据需要使用者自行配置；本仓库推荐的可运行入口见 [复现说明](../docs/reproduce.md)。

阅读当前实现时，从 `formal12/scripts/formal12_rl.py` 的 `choose` 与 `rollout_group` 开始，再看 `formal12_actor.py` 的 SFT 和更新，以及 `claude08_env.py` 的状态机。
