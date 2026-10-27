# YSZ 校准审核包

状态：待 Claude 审核，尚未提交。`01_calib/` 是未来远端同名目录的本地镜像；每项含 POSCAR/INCAR/KPOINTS/sub.sh/job.json/atom_mapping.json。这里不包含 POTCAR 原文或权重。

先看项目根目录 `CALIBRATION_REVIEW.md`、本目录 `JOB_LIST.csv` 和 `BUDGET.json`。仅在用户转交审核通过之后执行首波4项；ID不是执行顺序。不要直接批量执行所有sub.sh；当前阶段也不要在Mac执行它们。

`geometries/` 保存本机冻结MACE准备的初始/末态；`geometry_audit.json` 区分初始与末态占位；`CONFIGURATIONS.csv` 是对应索引。后续独立DFT终评不能重复使用本批校准标签冒充未见测试。
