"""Render Claude05's relative-energy validation and GPU recovery from current evidence."""
import datetime,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'
a=json.loads((C/'P0_ANALYSIS.json').read_text());b=json.loads((C/'BUDGET.json').read_text())
ga=json.loads((G/'ANALYSIS.json').read_text());gb=json.loads((G/'GPU_BUDGET.json').read_text())
s=json.loads((C/'REVIEW_STATUS.json').read_text());now=datetime.datetime.now(datetime.timezone.utc).isoformat()
ref=next((r for r in a['attempts'] if r.get('logical_job')=='ref_L00'),None)
gpu=[j for j in ga['jobs'] if j.get('approval_reference')=='REVIEW_CLAUDE_05_fast_gpu.md']
gate=a.get('Claude05_relative_energy_gate',{'status':'AWAITING_REFERENCE','passed':False})
lines=['# Claude05：相对能参照与GPU恢复实测','',f'更新：{now}。来源等级：原始文件/调度/解析为`code`；解释与预算建议为`inference`。',
       '', '以[REVIEW_CLAUDE_05_fast_gpu.md](REVIEW_CLAUDE_05_fast_gpu.md)为当前授权。P1/P2及以后暂停；仅09和07r立即放行，08r须07r完成且SCF独立检查通过再提交。不自动重试。','',
       '## CPU参照09 [code]','',
       '| 作业 | 状态 | CPU核 | 墙钟s | 核时 | SCF步 | SCF通过 | 每步平均s |','|---|---|---:|---:|---:|---:|---|---:|']
if ref:
 acc=ref['accounting'];v=ref.get('vasp',ref.get('partial_vasp',{}));mean=v.get('SCF_loop_real_seconds_mean')
 lines.append(f'| {ref["slurm_job_id"]}/09_L00_ref | {acc.get("State","未知")} | 128 | {acc.get("ElapsedRaw",0)} | {float(acc.get("CPUTimeRAW") or 0)/3600:.6f} | {v.get("n_scf_steps",v.get("last_printed_scf_step","待开始"))} | {v.get("SCF_converged",False)} | '+(f'{mean:.4f} |' if mean is not None else '待计算 |'))
 if acc.get('State')=='COMPLETED' and v.get('SCF_converged'):
  lines.insert(5,'**本批结束：09已收敛，相对能与力判据'+('通过' if gate.get('passed') else '未通过')+'；单GPU已通过CPU一致性，双GPU因MPI槽位启动问题未完成。P1/P2保持暂停，自动跟进已结束/暂停。**\n')
lines+=['','设置：vasp_std、Γ点、520eV、NBANDS1536、NCORE4、LREAL=.FALSE.、EDIFF=1E-6、AMIX=0.2、BMIX=0.0001。POSCAR和原L00完全相同。128核/192000MiB/1.5小时，单独192核时上限；不是重新放开原失败作业的自动重试。','',
        '## fast的相对能判据 [code / 用户授权]','',
        '相对能均为同成分、同晶胞下E(L01)−E(L00)。参考取09_L00_ref与旧L01/1882470；fast取05f/1882635与06f/1882636。',
        '门槛：相对能误差≤0.02eV/胞，并且L00、L01两组中最大逐原子力差均≤0.02eV/Å。旧的0.07598022eV/胞绝对偏移只作诊断，不再作为失败判据。','']
if gate.get('status')=='CHECKED':
 lines += [f'- ΔE_ref={gate["reference_relative_energy_eV"]:.9f}eV/胞。',
           f'- ΔE_fast={gate["fast_relative_energy_eV"]:.9f}eV/胞。',
           f'- 两者差的绝对值={gate["absolute_relative_energy_error_eV_cell"]:.9f}eV/胞。',
           f'- 两构型最大逐原子力差={gate["max_atom_force_difference_eV_A"]:.9f}eV/Å。',
           f'- 当前两构型健全性检查：**{"通过" if gate["passed"] else "未通过"}**。']
else:lines.append('09参考结果尚未完成/通过SCF，暂不判定相对能与力的误差。')
lines+=['','这只覆盖两个构型，不推广为整个排布空间的精度结论。P1后续三组Auto/.FALSE.配对仍待Claude放行。','',
        '## GPU恢复 [code]','',
        '| 作业 | GPU | MPI×线程 | 状态 | 墙钟s | 卡时 | SCF步/收敛 | 显存采样峰值MiB/卡 |','|---|---:|---|---|---:|---:|---|---:|']
for j in gpu:
 acc=j['accounting'];v=j.get('vasp',{})
 lines.append(f'| {j["slurm_job_id"]}/{j["name"]} | {j["gpus"]} | {j["mpi_ranks"]}×6 | {acc.get("State","未知")} | {acc.get("ElapsedRaw",0)} | {j["actual_card_hours"]:.9f} | {v.get("n_scf_steps","待完成")}/{v.get("SCF_converged",False)} | {j.get("sampled_max_GPU_memory_MiB","未采到")} |')
if not any(j['name'].startswith('08r') for j in gpu):lines.append('| 08r尚未提交 | 2 | 2×6 | 等07r完成并通过SCF | — | 0 | — | — |')
for j in gpu:
 p=j.get('cuda_preflight',{})
 if p:lines+=['',f'{j["name"]}预检：`{p.get("source")}`；通过={p.get("passed")}，GPU数={p.get("visible_GPU_count")}，真实驱动路径：'+', '.join(p.get('loaded_driver_paths',[]))+'。']
 comp=j.get('comparison_to_CPU_06f')
 if comp:lines+=['',f'{j["name"]}减CPU06f：总能差{comp["delta_E0_eV"]:.9f}eV/胞，最大逐原子力差{comp["max_atom_force_difference_eV_A"]:.9f}eV/Å；门槛分别0.01eV/胞和0.02eV/Å，**{"通过" if comp["passed"] else "未通过"}**；本次墙钟相对CPU06f快{comp["wall_speedup_over_CPU"]:.3f}倍。']
 if j.get('runtime_diagnostics',{}).get('MPI_insufficient_slots'):
  lines+=['',f'**{j["name"]}启动失败**：OpenMPI报告`There are not enough slots available ... satisfy the 2 slots`，尚未进入VASP/SCF。平台强制Slurm单任务12CPU，OpenMPI继承1个槽位；显式`--host ...:2`仍未覆盖Slurm槽位限制。CUDA预检识别2GPU且成功，当前是MPI启动问题。',
          '修复草案添加`--oversubscribe`只放宽MPI槽位计数，资源仍是原12CPU/2GPU；在同一个已分配作业内先用2个轻量Python进程验证各绑定6个互不重叠CPU、共12CPU，再允许VASP。草案只保存在本机，未提交；遵守“不自动重试”，等待新放行。']
lines+=['','驱动修复先过滤stubs和hpc_sdk CUDA lib64；只有过滤后cuInit失败才用LD_PRELOAD真实驱动。预检失败即退出，不启动VASP。平台强制1个Slurm任务/节点，2卡任务仍只分配12CPU，再在此分配内启动2个MPI进程×6线程；实际MPI/GPU数量以运行日志核验。','',
        '## 分池预算 [code]','',
        f'- CPU含冒烟、所有历史失败与本次09：实际{b["cumulative_actual_core_hours"]:.6f}核时；09剩余封顶{b.get("remaining_reference_core_hour_caps",192):.6f}；94原子待审预留1904；累计实耗加全部预留{b["actual_plus_remaining_caps"]:.6f}≤3000核时。',
        f'- GPU含547的4秒失败：实际{gb["actual_card_hours"]:.9f}卡时；已提交剩余加未提交08r预留{gb["remaining_card_hour_caps"]:.9f}；合计{gb["actual_plus_remaining_caps"]:.9f}卡时。审核的名义总封顶2.9511是原精确值2.951111111的四舍五入；脚本3480秒内部截止留足清理时间，实际按更严的2.9511监控，并且始终≤原批3卡时。',
        '- 09是已明确放行的316原子CPU校准例外。其余后续316原子走GPU规划，94原子走CPU；生产成本待有效GPU数据，不自动折算核时/卡时。','',
        '当前是否提交P1：**否，待Claude审核**。管理员反馈是附带建议，用户询问后暂放下，未发送任何对外消息。','',
        '原始证据：[CPU逐SCF耗时](calibration/SCF_TIMINGS.csv)、[GPU逐SCF耗时](calibration_gpu/SCF_TIMINGS.csv)、[CPU分析](calibration/P0_ANALYSIS.json)、[GPU分析](calibration_gpu/ANALYSIS.json)。',
        '终评独立复核：[四个CPU点的原始XML/EDIFF/输入哈希核对](calibration/evidence/Claude05_final_independent_verification.json)；待审[双GPU修复草案](calibration_gpu/recovery_draft/08s_L01_2g_slotsfix/DRAFT.json)。']
(ROOT/'CLAUDE05_RESULTS.md').write_text('\n'.join(lines)+'\n')
budget_lines=['# CPU/GPU分池预算：Claude05修订','',f'更新：{now}。预留不构成提交授权。','',
 '| 池子 | 当前授权 | 已消耗 | 本批未完成预留 | 后续待审规划 |','|---|---:|---:|---:|---|',
 f'| CPU | 3000核时 | {b["cumulative_actual_core_hours"]:.6f}核时 | 09剩余{b.get("remaining_reference_core_hour_caps",192):.6f}核时 | 94原子23项1904核时 |',
 f'| A100 | 原3卡时，审核05本批封顶约2.9511 | {gb["actual_card_hours"]:.9f}卡时 | {gb["remaining_card_hour_caps"]:.9f}卡时（含条件08r） | 316原子生产与后续校准待审 |','',
 f'CPU实耗加本批与后续预留={b["actual_plus_remaining_caps"]:.6f}核时，剩余{3000-b["actual_plus_remaining_caps"]:.6f}核时。GPU实耗与剩余预留合计={gb["actual_plus_remaining_caps"]:.9f}卡时；不加到CPU核时里。','',
 '94原子原待审清单：11单点×16=176；6收敛点×32=192；4个MACE起步+2个理想起步完整弛豫×256=1536，共1904。P1设置及新增3组Auto/.FALSE.配对要另列清单复核；当前没有放行。','',
 '09是本次明确批准的316原子CPU参照，封顶192；未来316原子CPU的22/23/30三项768核时已从原CPU预留移出，拟转GPU但未批准。Gamma计算用vasp_gam；k222收敛对照需vasp_std。','',
 '约2909卡时是账户余额快照，不是本任务生产授权；原5万CPU核时不能直接换算GPU卡时。约80个终评构型是否完整弛豫保持原研究设计，未擅自改为仅单点；成本需在GPU有效试算和后续pilot后估计。','',
 '最新结果：[CLAUDE05_RESULTS.md](CLAUDE05_RESULTS.md)；账本：[CPU](calibration/BUDGET.json)、[GPU](calibration_gpu/GPU_BUDGET.json)。']
(ROOT/'BUDGET_REPLAN.md').write_text('\n'.join(budget_lines)+'\n')
print(ROOT/'CLAUDE05_RESULTS.md')
