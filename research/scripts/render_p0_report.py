"""Render a reviewable P0 report from observed outputs only."""
from pathlib import Path
import datetime,json
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration'
a=json.loads((C/'P0_ANALYSIS.json').read_text());b=json.loads((C/'BUDGET.json').read_text())
status=json.loads((C/'REVIEW_STATUS.json').read_text())
claude03=status.get('approval_reference')=='REVIEW_CLAUDE_03_P0.md'
claude05=status.get('approval_reference')=='REVIEW_CLAUDE_05_fast_gpu.md'
claude04=status.get('approval_reference') in ['REVIEW_CLAUDE_04_L00.md','REVIEW_CLAUDE_05_fast_gpu.md']
summary=('Claude03已批准不再复测16核；三个64核大胞运行/核验中，P1按成本门槛决定，P2及以后待审。' if claude03 else
         '16核小胞超时，已停止扩批，等待 Claude 复核；32核小胞有效。' if status.get('p0_state')=='HALTED_AFTER_SMALLCELL_TIMEOUT' else
         '五项计算均已通过电子收敛检查，等待 Claude 复核。' if a['P0_all_five_converged'] else 'P0运行/核验中，结果尚未齐全。')
if claude03 and a.get('P0_reviewed_required_points_converged'):
    summary='Claude03要求的有效P0点已完成（16核已退役）；P1'+('已全部完成，P2及以后待审。' if a.get('P1_all_eleven_converged') else '已按条件提交，运行/核验中；P2及以后待审。' if (C/'P1_SUBMISSIONS.json').exists() else '是否放行见下方成本门槛，P2及以后待审。')
if status.get('p0_state')=='CLAUDE03_HALTED_BRMIX_L00':
    summary='L00连续BRMIX错误，已中止；未启动Auto已取消。L01在原授权限额内运行/待回收；P1及以后停止扩批，等待复核。'
if claude04:
    summary='Claude04的两项fast对照已授权/提交，旧L01自然结束；P1及以后明确暂停。'+('两项fast已通过SCF，待Claude审核。' if a.get('fast_both_converged') else 'fast运行/核验中。')
if claude05:
    summary='Claude05已放行09_L00_ref与GPU07r，改以相对能误差≤0.02eV/胞、最大逐原子力差≤0.02eV/Å检查；P1/P2保持暂停。'
logical={}
for row in a['attempts']:logical[row.get('logical_job',row['name'])]=row
lines=['# YSZ P0 实测报告','',
       f'更新：{a["analyzed_at_utc"]}（UTC）。来源等级：调度记录、原始输出和解析结果均为 `code`；建议和后续解释为 `inference`。','',
       '**状态：'+summary+'**','',
       '## 五个获批物理计算点','',
       '| 计算点 | 当前作业/状态 | CPU核 | 实际墙钟 s | 核时 | SCF步/是否收敛 | Slurm最大任务RSS | E0 eV/胞 | 最大力 eV/Å |',
       '|---|---|---:|---:|---:|---|---|---:|---:|']
base={j['id']:j for j in json.loads((C/'jobs.json').read_text())}
for name in ['01_S00_sp16','02_S00_sp32','03_L00_gamma','03b_L00_gamma_lrealauto','04_L01_gamma']:
    r=logical[name];acc=r['accounting'];v=r.get('vasp',{})
    rss=[s.get('MaxRSS','') for s in r['step_accounting'] if s.get('JobName')=='vasp_std']
    memory=','.join(rss) if rss else '未获得有效采样'
    invalid='不可用（未收敛）' if 'partial_vasp' in r else '未计算' if acc.get('State','').startswith('CANCELLED') else '待核验'
    e=f'{v["energy_eV"]["e_0_energy"]:.9f}' if v else invalid
    force=f'{v["max_force_eV_A"]:.5f}' if v else invalid
    scf=f'{v["n_scf_steps"]}/{"是" if v["SCF_converged"] else "否"}' if v else f'{r["partial_vasp"]["last_printed_scf_step"]}/否（截断）' if 'partial_vasp' in r else '未运行' if acc.get('State','').startswith('CANCELLED') else '待核验'
    elapsed=acc.get('ElapsedRaw','0');used=float(acc.get('CPUTimeRAW') or 0)/3600
    lines.append(f'| {name} | {r["slurm_job_id"]} / {acc.get("State","未知")} | {r.get("cores",base[name]["cores"])} | {elapsed} | {used:.6f} | {scf} | {memory} | {e} | {force} |')
lines+=['','SCF指电子自洽迭代；这里只验证单点电子收敛，未做离子弛豫。核时只汇总主作业行，避免把 batch、extern、MPI 步骤重复收费。内存栏是 Slurm 的最大任务 RSS，**不是整个作业所有进程的总峰值**。短任务的0/空值表示未有效采样，不能解释为没用内存。','',
        '## 启动排障及额外用量','',
        '最初 `mpirun` 经 Intel Hydra 只生成1个CPU的代理步骤。按核申请内存时，它又只继承1500M，而父作业申请的是24000/48000M，导致启动OOM。只改整节点内存解决了OOM，但保留1CPU限制造成异常慢速，因此两个计时中止。最终改用平台已有的 `/opt/hpc/slurm/21.08.6/lib/libpmi2.so` 与 `srun --mpi=pmi2`，步骤核数/任务数实际为16/16和32/32。','',
        '修复保持 POSCAR/INCAR/KPOINTS、总申请核数和内存不变；不把错误绑定的运行当性能测量。小胞重跑墙钟分别缩为20/25分钟，所有诊断用量计入原P0的216核时封顶。没有自动重新排队或循环重试。','',
        '| 尝试 | 作业 | 状态 | 墙钟 s | 实际核时 |','|---|---|---|---:|---:|']
for r in a['attempts']:
    if logical[r.get('logical_job',r['name'])]['slurm_job_id']==r['slurm_job_id']:continue
    acc=r['accounting'];lines.append(f'| {r["name"]} | {r["slurm_job_id"]} | {acc.get("State","未知")} | {acc.get("ElapsedRaw",0)} | {float(acc.get("CPUTimeRAW") or 0)/3600:.6f} |')
if (C/'evidence/01p_limit_amendment.json').exists():
    amendment=json.loads((C/'evidence/01p_limit_amendment.json').read_text())
    if amendment['command_exit_code']:
        lines+=['','16核计算偏慢，曾在原30分钟审定范围和216核时以内请求将20分钟改为29分钟；平台拒绝修改运行中限时，**未生效，实际仍为20分钟**。原始记录：[限时请求](calibration/evidence/01p_limit_amendment.json)。']
lines+=['','原始证明：[内存诊断](calibration/evidence/memory_diagnosis_stdout.txt)、[CPU诊断](calibration/evidence/cpu_affinity_diagnosis.txt)、[PMI2运行证明](calibration/evidence/pmi2_running_evidence.txt)、[完整尝试清单](calibration/P0_SUBMISSIONS.json)。启动方式依据 [Slurm官方Intel MPI说明](https://slurm.schedmd.com/mpi_guide.html#intel_mpi)。','',
        '## 16核与32核，以及 LREAL 对照','']
for key,title in [('16_vs_32','小胞16/32核'),('LREAL_False_vs_Auto','大胞LREAL=.FALSE./Auto')]:
    if key not in a['comparisons']:
        reason='16核结果截断，本轮无法比较。' if status.get('p0_state')=='HALTED_AFTER_SMALLCELL_TIMEOUT' and key=='16_vs_32' else '大胞未运行，无法判断。' if status.get('p0_state')=='HALTED_AFTER_SMALLCELL_TIMEOUT' else '待两个有效结果齐全，目前无法判断。'
        lines.append(f'- {title}：{reason}')
    else:
        x=a['comparisons'][key]
        lines.append(f'- {title}：后者减前者 E0={x["delta_E0_eV"]:.9f} eV/胞；力分量差RMSE={x["force_component_RMSE_eV_A"]:.6g} eV/Å，最大逐原子力差={x["max_atom_force_difference_eV_A"]:.6g} eV/Å；后者墙钟加速倍数={x["Slurm_wall_speedup_first_over_second"]:.3f}，核时比={x["core_hour_ratio_second_over_first"]:.3f}。')
lines+=['','各设置只有一次运行，节点和共享负载可能不同，速度差是本次观测。L00没有L01的Auto配对，**不能**从L00绝对总能差推出构型间相对能差误差；后续阶段采用哪种LREAL仍需复核。','',
        '## 预算与下一道检查','',
        f'- 旧P0所有尝试累计实际：**{b.get("actual_p0_core_hours",b["actual_new_job_core_hours"]):.6f} 核时**；P1实际 **{b.get("actual_p1_core_hours",0):.6f} 核时**；fast实际 **{b.get("fast_actual_core_hours",0):.6f} 核时**；09参照实际 **{b.get("reference_actual_core_hours",0):.6f} 核时**；含已有冒烟合计：**{b["cumulative_actual_core_hours"]:.6f} 核时**。',
        f'- P0实际加未完成点预留封顶：**{b["p0_actual_plus_remaining_caps"]:.6f} ≤216 核时**。',
        f'- 加所有未完成点预留及冒烟：**{b["actual_plus_remaining_caps"]:.6f} ≤3000 核时**。',
        '- 原31项清单的2888核时只保留为历史规划；现执行上限以最新分阶段账本为准。失败和诊断按实耗记账，不把取消的原任务和替代任务同时预扣。预留不构成重算或扩批授权。',
        '- P2及后续保持暂停；31_S02/32_S06理想格点起步跨位诊断已准备，尚未执行。P0不能回答跨位是否真实，也不能证明MACE总体排序准确。','',
        '机器可读：[BUDGET.json](calibration/BUDGET.json)、[分析结果](calibration/P0_ANALYSIS.json)、[最新调度快照](calibration/P0_LATEST.json)、[回收的原始输出](calibration/P0_outputs)。']
if claude04:
    lines+=['','## Claude04：两项fast对照 [code / inference]','',
            '授权原件：[REVIEW_CLAUDE_04_L00.md](REVIEW_CLAUDE_04_L00.md)。05f_L00、06f_L01各128核、192000 MiB、1小时，不自动重排或重试。只有Γ点，使用vasp_gam；LREAL=Auto、EDIFF=1E-5、AMIX=0.2、BMIX=0.0001，其余物理输入及POSCAR与对应旧作业相同。',
            '两项一起提交；旧04仍运行时，06f只等旧04自然结束（afterany），05f可先运行，使本项目始终最多2个作业并发。旧04即使超时也不按旧设置重跑。P1旧的≤40核时自动放行规则已作废；P1及以后保持暂停。','',
            '| 点 | 作业/状态 | CPU | 墙钟 s | 核时 | SCF步/收敛 | 每步平均 s | E0 eV/胞 |','|---|---|---:|---:|---:|---|---:|---:|']
    for name in ['04_L01_gamma','fast_L00','fast_L01']+(['ref_L00'] if claude05 else []):
        r=logical.get(name)
        if not r:continue
        acc=r['accounting'];v=r.get('vasp',{});p=r.get('partial_vasp',{})
        steps=f'{v["n_scf_steps"]}/是' if v.get('SCF_converged') else f'{p.get("last_printed_scf_step","待开始")}/未核验'
        mean=v.get('SCF_loop_real_seconds_mean',p.get('SCF_loop_real_seconds_mean'))
        mean_text=f'{mean:.4f}' if mean is not None else '待计算'
        energy=f'{v["energy_eV"]["e_0_energy"]:.9f}' if v.get('SCF_converged') else '不可作能量标签'
        lines.append(f'| {name} | {r["slurm_job_id"]}/{acc.get("State","待调度") } | {r.get("cores",64)} | {acc.get("ElapsedRaw",0)} | {float(acc.get("CPUTimeRAW") or 0)/3600:.6f} | {steps} | {mean_text} | {energy} |')
    lines+=['','所有完整SCF步墙钟：[SCF_TIMINGS.csv](calibration/SCF_TIMINGS.csv)。运行中部分输出仅供进度诊断，不用作能量或力标签。','',
            f'fast单独授权封顶256核时；本次实际加未完成预留为{b.get("fast_actual_plus_remaining_caps",256):.6f}核时。旧P0的216封顶与fast的256分开记账，全部校准仍≤3000。'+('用户后续选B，未来CPU只保留94原子的23项待审计算，封顶1904核时；316原子3项后续校准转GPU规划，但尚未放行。新预算见[BUDGET_REPLAN.md](BUDGET_REPLAN.md)。' if b.get('cpu_future_scope')=='94_ATOM_ONLY_AFTER_GPU_SELECTION' else '新增256从尚未放行P2+预留中调拨，P2+预留2240，后续清单需重排再审；没有追加在原计划之上。'),'',
            '### 新旧L01精度与成本对照','']
    comp=a['comparisons'].get('L01_old_vs_fast')
    if comp:
        lines.append(f'fast减旧L01：E0差{comp["delta_E0_eV"]:.9f} eV/胞，TOTEN差{comp["delta_TOTEN_eV"]:.9f} eV/胞；最大逐原子力差{comp["max_atom_force_difference_eV_A"]:.6g} eV/Å，力分量RMSE{comp["force_component_RMSE_eV_A"]:.6g} eV/Å；墙钟加速{comp["Slurm_wall_speedup_first_over_second"]:.3f}倍，核时比{comp["core_hour_ratio_second_over_first"]:.3f}。')
        lines.append(('审核04的绝对能差指标保留为诊断，**审核05已改用相对能门槛，不再以此绝对偏移判失败**。' if claude05 else '绝对能差≤0.02eV/胞检查：**'+('通过' if abs(comp['delta_E0_eV'])<=.02 else '未通过')+'**。')+'只测一个同构型，无法判断此偏移是否在构型相对能中抵消；没有自动采用fast作为生产协议。')
    else:
        lines.append('只有旧L01和fast L01都正常结束且SCF通过才计算误差；若旧L01超时，这一精度对照无法判断，本轮不额外重跑。')
    comp=a['comparisons'].get('L01_minus_L00_fast')
    lines+=['','### fast协议的构型相对能','',
            f'E0(L01)−E0(L00)={comp["delta_E0_eV"]:.9f} eV/316原子胞。' if comp else '等待两个fast构型均通过SCF；暂不计算相对能。','',
            'EDIFF是电子迭代停止阈值，除以原子数不能当作已经验证的总能误差。新旧L01对照测的是整套fast设置的合并影响，不能拆出单个参数的贡献。旧L00没有有效DFT能量，因此仅凭一个L01对照也无法量化构型间相对能偏差。',
            '电荷混合不稳是当前待验证诊断；本轮测试新参数是否解决，不能先保证一定修复。最短距离检查排除了明显原子重叠，但不替代势能面或跨位物理判断。','',
            ('用户后续已选B，并另外批准1/2GPU L01对照合计≤3卡时；CPU与GPU分别记账。GPU进展见[GPU_PILOT_RESULTS.md](GPU_PILOT_RESULTS.md)。P1/P2保持暂停；不把CPU fast收敛当作精度自动放行。' if (ROOT/'GPU_PILOT_PLAN.md').exists() else '主终评与GPU试算留待实测后另定；本批没有提交GPU、训练或跨位弛豫。'),
            '新输入与提交账：[CLAUDE04_JOB_LIST.json](calibration/CLAUDE04_JOB_LIST.json)、[FAST_SUBMISSIONS.json](calibration/FAST_SUBMISSIONS.json)。']
elif claude03:
    lines+=['','## Claude03放行规则与补充核实','',
            '- 原件：[REVIEW_CLAUDE_03_P0.md](REVIEW_CLAUDE_03_P0.md)。不再复测16核；03/03b/04各64核、1小时、96GB，三个作业合计192核时封顶，最多2并发。',
            '- P1须三个大胞均正常完成且SCF通过，并且03/04各自实际核时≤40；满足后11小胞以32核/30分钟/48GB、每项16核时执行。P2及以后仍待审。',
            '- 旧冒烟1880254的逐步账显示MPI步骤分配了16核，TotalCPU约292秒/23秒墙钟，不能说旧冒烟只跑1核。后续显式资源设置下Hydra子步骤的1核/1500M问题有独立账据。原冒烟模板已备份并改为PMI2，未重跑。证据：[旧冒烟复核](calibration/evidence/Claude03_preflight.txt)、[模板更新](calibration/evidence/smoke_template_update.json)。','',
            '### 每个电子步的耗时','',
            '完整逐步数据：[SCF_TIMINGS.csv](calibration/SCF_TIMINGS.csv)。下表均为OUTCAR中LOOP的rank0墙钟；实际计费仍按Slurm整作业分配核时。','',
            '| 计算点 | 作业 | 电子步数 | 每步平均 s | 每步中位数 s |','|---|---|---:|---:|---:|']
    for name in ['02_S00_sp32','03_L00_gamma','03b_L00_gamma_lrealauto','04_L01_gamma']:
        r=logical[name];v=r.get('vasp',r.get('partial_vasp',{}));mean=v.get('SCF_loop_real_seconds_mean');median=v.get('SCF_loop_real_seconds_median')
        steps=v.get('n_scf_steps',str(v['last_printed_scf_step'])+'（未收敛）' if 'last_printed_scf_step' in v else '待完成')
        lines.append(f'| {name} | {r["slurm_job_id"]} | {steps} | '+(f'{mean:.4f} | {median:.4f} |' if mean is not None else '待完成 | 待完成 |'))
    comp=a['comparisons'].get('L01_minus_L00_LREAL_False')
    lines+=['','### 同设置下L01相对L00','']
    lines.append(f'LREAL=.FALSE.、Γ、520eV下 E0(L01)−E0(L00)={comp["delta_E0_eV"]:.9f} eV/316原子胞。尚待P2做k点/截断收敛。' if comp else '两个大胞基准结果尚未齐全，暂不计算相对能。')
    gate=a.get('P1_release_gate',{})
    lines+=['','### P1条件判定','',f'当前允许提交：**{gate.get("allowed",False)}**。',json.dumps(gate,ensure_ascii=False,indent=2)]
    p1=[(name,logical.get(name)) for name,j in base.items() if j['phase']=='P1_small']
    if (C/'P1_SUBMISSIONS.json').exists():
        lines+=['','### P1批次','', '| 计算点 | 作业/状态 | 核时 | SCF | E0 eV/胞 |','|---|---|---:|---|---:|']
        for name,r in p1:
            if not r:continue
            acc=r['accounting'];v=r.get('vasp',{});energy=f'{v["energy_eV"]["e_0_energy"]:.9f}' if v else '待核验'
            lines.append(f'| {name} | {r["slurm_job_id"]}/{acc.get("State","未知")} | {float(acc.get("CPUTimeRAW") or 0)/3600:.6f} | {v.get("SCF_converged","待核验")} | {energy} |')
    if status.get('p0_state')=='CLAUDE03_HALTED_BRMIX_L00':
        lines+=['','### 本次停止原因与待复核项 [code / inference]','',
                'L00在SCF第5、6步连续报告 `BRMIX: very serious problems`，输出的电子数从设定2528变为2533.72、2565.10；第6步中间总能跳至约−178226 eV。已中止1882469，取消尚未启动的Auto1882471。L011882470未出现同类报错，仍限于已批64核/1小时。没有启动P1，没有重算。',
                '这些是未收敛电子迭代的数值，不能作为材料能量，也不能据此判断结构不稳定。根因尚未确认，需要先复核结构/电子初始化/数值设置，再设计受控修复；本轮不擅自更改设置试跑。',
                '原始诊断：[SCF6诊断](calibration/evidence/Claude03_SCF6_diagnostic.json)、[中止记录](calibration/evidence/Claude03_BRMIX_stop.txt)。自动跟进已按停止规则暂停；L01结果在远端写入，后续只需回收，不能重复提交。']
elif status.get('p0_state')=='HALTED_AFTER_SMALLCELL_TIMEOUT':
    result=logical['02_S00_sp32'].get('vasp',{})
    lines+=['','## 给 Claude 的复核问题 [inference]','',
            '1. 是否接受停止16核复测，保留32核这一份有效同几何结果，以32核继续原先获批的大胞03/03b/04？目前三个大胞都未运行。若仍需16核对照，须另定足够的限时和累计预算，不自动重跑。',
            '2. 这次16核在20分钟墙钟后被Slurm标为TIMEOUT（实际含停机约29秒），最后只打印到SCF第9步；不能使用截断能量，更不能与32核的最终能量做精度对照。',
            '3. 32核有效结果也只有一个小胞，尚不能放行MACE排序、主尺寸成本或LREAL设置。']
    if 'DFT_vs_cached_MACE_force_component_RMSE_eV_A' in result:
        lines+=['',f'本小胞DFT对缓存MACE的力分量RMSE={result["DFT_vs_cached_MACE_force_component_RMSE_eV_A"]:.6f} eV/Å，最大逐原子力误差={result["DFT_vs_cached_MACE_max_atom_force_error_eV_A"]:.6f} eV/Å；样本量1，只是诊断。']
if claude05:
    lines+=['','## Claude05最新相对能判据 [code / 用户授权]','',
            '新增09_L00_ref采用vasp_std、LREAL=.FALSE.、EDIFF=1E-6，仅修改AMIX/BMIX以稳定电荷混合；128核、1.5小时、独立192核时上限。与旧L01组合计算参考相对能。','',
            '门槛：|ΔE_fast(L01−L00)−ΔE_ref(L01−L00)|≤0.02eV/胞，两个构型中最大逐原子力差均≤0.02eV/Å；不是把每原子绝对偏移当作通过证据。','',
            '```json',json.dumps(a.get('Claude05_relative_energy_gate',{}),ensure_ascii=False,indent=2),'```','',
            'GPU恢复与总体结果见[CLAUDE05_RESULTS.md](CLAUDE05_RESULTS.md)。07r完成SCF才可提交08r，P1/P2始终等Claude放行。']
(ROOT/'P0_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(ROOT/'P0_RESULTS.md')
