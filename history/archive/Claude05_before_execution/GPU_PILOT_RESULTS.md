# A100 L01 实测：启动失败、已停止及恢复草案

更新：2026-10-23T12:19:37.745958+00:00。状态：547失败，548启动前取消；CPU参照完整回收；P1/P2及以后暂停，不自动重试。

## 实测和用量 [code]

| 项目 | 作业 | 状态 | 墙钟 | 实际用量 | 有效DFT能量 |
|---|---|---|---:|---:|---|
| 1×A100，1MPI×6线程 | 547 | CUDA初始化退出 | 4秒 | 0.001111111卡时 | 无，0个SCF步 |
| 2×A100，2MPI×6线程 | 548 | 启动前取消 | 0秒 | 0卡时 | 无 |
| CPU fast L00，128核 | 1882635 | 22步收敛 | 813秒 | 28.906667核时 | −2996.75904609eV |
| CPU fast L01，128核 | 1882636 | 21步收敛 | 837秒 | 29.760000核时 | −2997.46683384eV |
| CPU旧L01，64核 | 1882470 | 20步收敛 | 3538秒 | 62.897778核时 | −2997.39085362eV |

GPU试算总预算3卡时，本次实际0.001111111，未用额度2.998888889。CPU全部含失败/诊断/冒烟累计169.484444核时。两种单位、账户分开记录，不能相加。GPU未产生可比能量，GPU加速、1卡/2卡性价比、GPU与CPU能量/力一致性均**无法判断**。

GPU显存只来得及采到启动时0MiB；不能用来估计316原子计算所需显存。Slurm最大任务RSS是空/0，表示任务太短而没有有效采样。

## 根因证据 [code / direct_text]

07g日志：`Accelerator Fatal Error: call to cuInit returned error 34`。NVIDIA将错误34定义为`CUDA_ERROR_STUB_LIBRARY`，即加载了编译用占位驱动。[NVIDIA官方定义](https://nvidia.github.io/cuda-python/cuda-bindings/12.8.0/module/driver.html)

实际读取的共享目录存在以下两个符号链接：

```text
/data/software/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/lib64/libcuda.so
/data/software/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/lib64/libcuda.so.1
 -> /data/software/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so
```

管理员要求的nvhpc模块将这个目录加入LD_LIBRARY_PATH。GPU已分配、nvidia-smi已识别A100、三个赝势/二进制/物理输入哈希都通过；随后VASP在cuInit处退出，没有进入材料计算。证据在[驱动库诊断](calibration_gpu/evidence/CUDA_library_diagnosis.txt)、[原始输出](calibration_gpu/outputs/07g_L01_1g/log.txt)、[账本](calibration_gpu/GPU_BUDGET.json)。

另一个已解决的免费提交前问题：平台job_submit.lua强制一个Slurm任务/节点，2GPU需以1个12CPU分配单元，再在其内部启动2个MPI进程×6线程。原2任务/节点写法只在test-only阶段被拒绝，未产生付费作业。1卡实际日志已确认1MPI×6线程；2卡尚未运行，实际2MPI的验证仍待做。

## 准备好的恢复草案 [code / inference]

只在项目作业环境中优先加载计算节点真实libcuda.so.1；排除包含stub的路径，LD_PRELOAD只指定这一个真实驱动库。不改平台文件、不改全局模块、不改POSCAR/INCAR/KPOINTS。草案中先做cuInit预检，失败就退出、不启动VASP。

本机已准备[真实驱动选择脚本](calibration_gpu/recovery_draft/select_real_driver.sh)和07r/08r两个输入草案，Bash语法检查通过，物理输入字节一致。具体计算节点真实驱动路径尚未实测：无活动作业时，平台的pam_slurm_adopt阻止只读登录计算节点；没有绕过这一限制。不能声称修复已在GPU上验证。

建议恢复时两项墙钟均缩到59分钟，含已花4秒的累计硬上限为2.951111111卡时，小于3。草案未上传、未提交；恢复是否放行待用户/Claude审核，依据当前工作单明确要求“不自动重试”。[恢复草案清单](calibration_gpu/RECOVERY_DRAFT.json)

## CPU精度对照的重要结果 [code / inference]

相同L01：fast−旧协议总能为-0.075980220eV/胞，绝对差超过预设0.02eV/胞，**绝对能量一致性未通过**。最大逐原子力差0.006461165eV/Å，力分量RMSE0.001809305eV/Å。墙钟快4.227倍，核时减少52.69%。

fast下E(L01)−E(L00)=-0.707787750eV/胞；旧L00没有有效能量，不能判断fast偏移在构型差中能否抵消。0.07598eV是整套设置差的实测，不单独归因LREAL、EDIFF或硬件。GPU/CPU一致性≤0.01eV/胞即使未来通过，也不自动修复fast对旧协议的偏差。

用户粘贴中把总能差和最大力差一起写为≤0.01eV/胞；这个单位只适用于总能。力差按eV/Å报告，力的验收阈值需要明确，不能混用单位宣布通过。

## 预算重排和下一步

316原子后续转GPU规划，94原子后续留CPU；详见[BUDGET_REPLAN.md](BUDGET_REPLAN.md)。GPU速度未测出，尚不能把单点或完整弛豫卡时填成实测；P1/P2、训练及生产DFT均未放行。
