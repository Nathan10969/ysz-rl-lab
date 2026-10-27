# 集群资源只读核查

日期：2026-10-22（America/Los_Angeles）。通过现有 SSH 别名查询；未提交作业、未运行计算、未修改远端文件或其他项目。

## A100：已实际连通

- `sinfo`：每节点 48 CPU 核、8 张 A100；`scontrol show node gpu01`：2 socket × 24 cores，ThreadsPerCore=1。
- `/data/GPU-RESOURCE-RULE.txt` 原文：严格按 **1 GPU : 6 CPU : 90 GiB 内存** 分配，每作业最多 8 张卡；节点配 A100 40GB。
- 1/2/4/8 张 GPU 对应 6/12/24/48 CPU 核、90/180/360/720 GiB 内存。登录节点的核数不是作业额度。
- 查询瞬间 gpu01：1 张卡已分配、7 张未分配；gpu02 为 DRAIN，暂不可正常调度。空闲资源会变化，未在此保证立即分配。
- 两次初始连接曾报 `Exceeded MaxStartups`，之后相同别名成功；没有改 IP、关闭主机验证或更换密钥。

## CPU 超算：已实际连通

- 队列 `q_amd_share`，每计算节点 **128 物理 CPU 核**、约 193,288 MiB 内存；AMD EPYC 7H12，2 socket × 64 cores，ThreadsPerCore=1。
- 分区共列 240 节点、30,720 CPU 核，属于整个共享分区，不能当作个人额度。
- 仅查询本用户关联的 `sacctmgr` 结果（按字段）：GrpNodes=8，MaxJobs=12，GrpCPUs/MaxCPUs/MaxNodes 未显示数值，QOS=normal,standby。
- 这条关联的节点上限对应最多 8×128=1,024 核的硬件容量；上级账号/QOS/调度和余额仍可能限制实际可申请值，未检查完这些继承限制，不能称已保证可用的个人核数。
- `scontrol show node bn169` 交叉确认 CPUTot=128，ThreadsPerCore=1；即时空闲节点存在，但未申请。

## 对预算的意义

CPU 核时和 GPU 卡时分开记录。A100 作业配 6 个 CPU 核并不代表只具有 6 核 CPU 的算力，GPU 版计算还使用整张 A100；旧的 CPU 核时预算不能除以 6 来预测 GPU 作业耗时。按标准卡时单位，1 张 GPU 运行 10 小时是 10 卡时；平台是否还有 CPU/软件附加计费须另核实。尚未测速云端 VASP，也没有扣用计划中的校准计算额度。

查询命令：`sinfo -N`、`scontrol show partition`、`scontrol show node`、`lscpu`、仅针对本用户的 `sacctmgr show assoc`，以及只读上述平台资源规则和 `/data/about-hpc.txt`。
