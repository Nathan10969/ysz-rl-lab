> 以下保留审核04阶段的历史记录；最新审核05授权、执行状态和结果见 [CLAUDE05_RESULTS.md](CLAUDE05_RESULTS.md)。

# A100 L01 小规模对照：用户授权记录与执行方案

来源：2026-10-23用户在当前聊天直接转达“审核04补充 A100 VASP-GPU”，选B，批准最多3卡时。这是Codex整理的授权记录，不是另造的Claude审核原件。[用户授权]

- CPU的05f/06f继续原限额；新增GPU只算L01两次：07g_L01_1g=1GPU/1MPI/6CPU/90GiB/1h；08g_L01_2g=2GPU/2MPI/12CPU/180GiB/1h。后者afterok前者，便于失败停止和避免同节点竞争影响测速。[用户授权/inference]
- 目录hzw-a100:~/Jony/runs/01_calib_gpu/。PBE/POSCAR/KPOINTS/ENCUT520/NBANDS1536/EDIFF1E-5/LREALAuto/AMIX0.2/BMIX0.0001与CPU06f相同；GPU并行NCORE=1、KPAR=1，NSIM默认。vasp_gam，mkl/latest、nvhpc/25.3，每GPU一个MPI进程、每进程6线程。[code]
- 三个POTCAR逐个SHA与CPU完全一致；二进制与输入运行前校验；POTCAR原文不回传。脚本内部59分钟截止、10秒强制退出宽限，防止Slurm KillWait导致超过3卡时；不自动重试。[code]
- 比较每SCF步、SCF步数、总墙钟、实际GPU卡时、GPU显存采样，以及同L01对CPU06f的总能和最大逐原子力差。只有COMPLETED且SCF独立通过的输出可比较。能量一致要求绝对差≤0.01eV/胞；力差单位为eV/Å，用户粘贴的0.01eV/胞不能用于力，先如实报告，阈值待审核。[用户授权/inference]
- 1/2GPU性能同时看墙钟和GPU卡时，不能只选最快。通过精度后再重写316原子大胞GPU预算；94原子后续DFT归CPU预算。GPU余额2909是用户提供的快照；3卡时之外没有获批GPU开销。[inference]
- P1、P2及以后暂停，不提交训练、理想格点弛豫或生产DFT。失败先停止后继、保存实耗，不自动重试。[用户授权]

并行设置依据：[VASP官方GPU说明](https://vasp.at/wiki/OpenACC_GPU_port_of_VASP)、[MPI/OpenMP说明](https://vasp.at/wiki/Combining_MPI_and_OpenMP)；管理员/data/software/app/demo_vasp/run.slurm已读取。[direct_text/code]

## 平台提交器的实测约束 [code]

/etc/slurm/job_submit.lua强制每节点1个Slurm任务，并把cpus_per_task改为6×GPU数。原2任务/节点配置在不计费的test-only阶段被拒绝，没有提交实际GPU作业。现按平台要求分配1个12CPU的任务，再在同一已分配节点内明确启动2个MPI进程、每进程6线程；GPU数、总CPU、总内存、墙钟和物理输入不变。1卡测试同理为1个6CPU任务/1MPI。
