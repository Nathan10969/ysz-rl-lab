# 每日更新约定

## 时间与输入

用户授权每天 America/Los_Angeles 20:00，从本机研究工作区整理到本仓库并同步 GitHub。工作区路径只保存在 Git 忽略的 `.local-source.json`。

自动任务只读取已有记录和结果。公开包检查和日更都从本机执行，使用本机现有的研究文件与 GitHub 鉴权。

## 一次更新

```bash
python3 tools/publish_snapshot.py
python3 tools/results.py --write
python3 tools/check_publication.py
python3 -m unittest discover -s tests -v
git diff --stat
```

维护者或本机接续随后阅读变化：更新 README 的当前阶段、结果解释和当天日志，区分实测、解释与计划；核对后按真实日期提交推送。有新证据推翻旧判断时追加更正依据，并保留历史版本。

只使用明确列在 `tools/publication_sources.json` 中的原始文件。模型、虚拟环境、凭据和消息库不会因新增文件自动进入白名单。结果摘要通过字段级导出。

菜单偏好读数单独保留在结果摘要的 `menu_preference` 字段，由 `results.py --write` 复算选参门槛并生成 `experiments/diagnostics/menu_grid_summary.json`。C 不混入能量排名。账本表头不一致时保留原值并输出 `header_totals_consistent=false`，不能把报告总额当作已结算费用。

`publish_snapshot.py` 只要输入没有变化就保持文件不变，不会靠修改时间戳制造日更。一天多次同步会追加记录，不删除已经写下的观察。

## 失败和等待

导出、校验或推送失败时保留本地文件与真实错误，停止推送；如果需要用户处理，删除这条每日自动接续并说明原因。完成日更这项持续安排前，遵循用户后来发出的修改／停止指示。

## 如何在另一台机器使用导出工具

```bash
python3 tools/publish_snapshot.py --source-root /path/to/your/research-workspace
```

它不执行 SSH、训练、MACE、DFT 或调度命令。历史运维代码只是供研究追溯，日更工具不会调用它们。
