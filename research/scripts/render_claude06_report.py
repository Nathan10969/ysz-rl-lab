"""Render live Claude06 scientific checks; pending results never become labels."""
import datetime,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from ase.geometry import find_mic
from ase.io import read
from calibration import reference,canonical_key,shell_counts
from claude06_safety import TERMINAL
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'

def load(path,default):return json.loads(path.read_text()) if path.exists() else default
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def with_forces(config,energy,forces,xyz):
    mace=read(xyz);diff=np.asarray(forces)-mace.get_forces()
    return dict(config=config,DFT_energy_eV=float(energy),MACE_energy_eV=float(mace.get_potential_energy()),
                force_component_RMSE_eV_A=float(np.sqrt(np.mean(diff**2))),
                atom_force_errors_eV_A=np.linalg.norm(diff,axis=1).tolist())

def gate(rows,expected):
    result=dict(n_valid=len(rows),n_required=expected,status='AWAITING_DATA',passed=None,
                top_rule='MACEtop3 intersectsDFTtop6 at least2; also reporttop3intersection')
    if len(rows)!=expected:return result
    me=np.array([r['MACE_energy_eV'] for r in rows]);de=np.array([r['DFT_energy_eV'] for r in rows])
    correlation=float(spearmanr(me,de).statistic)
    top_mace=np.argsort(me)[:3];top_dft3=np.argsort(de)[:3];top_dft6=np.argsort(de)[:6]
    retain=len(set(top_mace)&set(top_dft6));median=float(np.median([r['force_component_RMSE_eV_A'] for r in rows]))
    p95=float(np.quantile(np.concatenate([r['atom_force_errors_eV_A'] for r in rows]),.95))
    result.update(status='CHECKED',spearman=correlation if np.isfinite(correlation) else None,
        top3_retained_in_DFT_top6=retain,top3_intersection=len(set(top_mace)&set(top_dft3)),
        force_component_RMSE_median_eV_A=median,atom_force_error_p95_eV_A=p95,
        MACE_top3=[rows[i]['config'] for i in top_mace],DFT_top6=[rows[i]['config'] for i in top_dft6],
        passed=bool(np.isfinite(correlation) and correlation>=.7 and retain>=2 and median<=.2 and p95<=.5),
        per_configuration=[dict(config=r['config'],force_component_RMSE_eV_A=r['force_component_RMSE_eV_A'],
              atom_force_error_p95_eV_A=float(np.quantile(r['atom_force_errors_eV_A'],.95)),
              max_atom_force_error_eV_A=max(r['atom_force_errors_eV_A'])) for r in rows])
    return result

def occupancy(atoms,parent,initial):
    mapped=parent.copy();mapped.numbers[:]=0;assigned=np.full(len(atoms),-1,dtype=int);distance=np.zeros(len(atoms))
    for mask,sites in [(np.isin(atoms.numbers,[39,40]),np.flatnonzero(parent.numbers==40)),(atoms.numbers==8,np.flatnonzero(parent.numbers==8))]:
        ai=np.flatnonzero(mask);v=atoms.positions[ai,None,:]-parent.positions[sites][None,:,:]
        dist=find_mic(v.reshape(-1,3),parent.cell,pbc=True)[1].reshape(len(ai),len(sites))
        ii,jj=linear_sum_assignment(dist**2);assigned[ai[ii]]=sites[jj];distance[ai[ii]]=dist[ii,jj]
    assert len(set(assigned))==len(atoms) and np.all(assigned>=0)
    mapped.numbers[assigned]=atoms.numbers;n=2 if len(parent)==96 else 3
    return dict(canonical_key=canonical_key(mapped,n),occupation_class_changed=canonical_key(mapped,n)!=canonical_key(initial,n),
       vacancy_site_ids=np.flatnonzero((mapped.numbers==0)&(parent.numbers==8)).tolist(),
       Y_site_ids=np.flatnonzero(mapped.numbers==39).tolist(),max_assignment_distance_A=float(distance.max()),
       assigned_site_ids=assigned.tolist(),shell_counts=shell_counts(mapped))

def main():
    cpu=load(C/'CLAUDE06_ANALYSIS.json',{'jobs':[]})['jobs'];gpu=load(G/'CLAUDE06_ANALYSIS.json',{'jobs':[]})['jobs']
    good={r['name']:r for r in cpu+gpu if r['usable']}
    small=[];large=[]
    for r in good.values():
        if r['phase'] not in ['P1_small','P2b_maincell']:continue
        row=dict(config=r['config'],DFT_energy_eV=r['energy_eV']['e_0_energy'],MACE_energy_eV=r['MACE_energy_eV'],
                 force_component_RMSE_eV_A=r['MACE_force_component_RMSE_eV_A'],atom_force_errors_eV_A=r['MACE_atom_force_errors_eV_A'])
        (small if r['phase']=='P1_small' else large).append(row)
    historical=load(C/'P0_ANALYSIS.json',{'attempts':[]})
    old=next((r for r in historical['attempts'] if r['name']=='02p_S00_sp32' and r.get('vasp',{}).get('SCF_converged')),None)
    small12=list(small)
    if old:
        v=old['vasp'];small12.append(with_forces('S00',v['energy_eV']['e_0_energy'],v['forces_eV_A'],C/'geometries/S00_relaxed.extxyz'))
    pair=[]
    for cid in ['S01','S04','S08']:
        a=next((r for r in cpu if r['config']==cid and r['phase']=='P1_small' and r['usable']),None)
        b=next((r for r in cpu if r['config']==cid and r['phase']=='P1_lreal_pair' and r['usable']),None)
        if a and b:pair.append(dict(config=cid,Auto_minus_False_eV=a['energy_eV']['e_0_energy']-b['energy_eV']['e_0_energy'],
                   max_atom_force_difference_eV_A=float(np.linalg.norm(np.array(a['forces_eV_A'])-np.array(b['forces_eV_A']),axis=1).max())))
    pair_gate=dict(n_pairs=len(pair),rows=pair,passed=None)
    if len(pair)==3:
        spread=float(np.ptp([r['Auto_minus_False_eV'] for r in pair]));force=max(r['max_atom_force_difference_eV_A'] for r in pair)
        pair_gate.update(energy_offset_spread_eV=spread,max_atom_force_difference_eV_A=force,passed=spread<=.02 and force<=.02)
    convergence=[]
    for variant in ['encut650','k333']:
        values={}
        for cid in ['S00','S04','S08']:
            hi=next((r for r in cpu if r['config']==cid and r['name'].endswith(variant) and r['usable']),None)
            lo=next((r for r in cpu if r['config']==cid and r['phase']=='P1_small' and r['usable']),None)
            if cid=='S00' and old:low=old['vasp']['energy_eV']['e_0_energy']
            elif lo:low=lo['energy_eV']['e_0_energy']
            else:low=None
            if hi and low is not None:values[cid]=hi['energy_eV']['e_0_energy']-low
        if len(values)>=2:
            spread=float(np.ptp(list(values.values())))
            convergence.append(dict(variant=variant,n_configs=len(values),per_config_high_minus_base_eV=values,
                relative_energy_change_span_eV=spread,within_0_02_eV=spread<=.02,
                caveat='S00baseFalse/e6;S04S08baseAuto/e5; inspectsameprotocolS04-S08separately',
                S04_S08_same_protocol_relative_change_eV=abs(values['S04']-values['S08']) if {'S04','S08'}<=set(values) else None))
    cross=[];cost=[]
    for r in gpu:
        if r['kind']!='relax':continue
        cost.append(dict(name=r['name'],config=r['config'],state=r['state'],actual_card_hours=r['actual'],
                         ionic_converged=r.get('convergence',{}).get('ionic_converged'),n_ionic_outputs=r.get('convergence',{}).get('n_ionic_outputs')))
        if not r['usable']:continue
        cid=r['config'];initial=read(C/'geometries'/f'{cid}_initial.extxyz');parent=reference(2 if r['n_atoms']==94 else 3)
        final=read(G/'claude06_outputs'/r['name']/'CONTCAR',format='vasp');mace=read(C/'geometries'/f'{cid}_relaxed.extxyz')
        dftmap=occupancy(final,parent,initial);macemap=occupancy(mace,parent,initial)
        cross.append(dict(name=r['name'],from_ideal=r['input_geometry']=='ideal_initial_occupation',DFT=dftmap,MACE=macemap,
                          same_MACE_DFT_occupation_class=dftmap['canonical_key']==macemap['canonical_key']))
    budget=load(C/'BUDGET.json',{});gb=load(G/'GPU_BUDGET.json',{})
    all_terminal=len(cpu)==20 and len(gpu)==25 and all(r['state'] in TERMINAL for r in cpu+gpu)
    all_usable=all_terminal and all(r['usable'] for r in cpu+gpu)
    result=dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),approval_reference='REVIEW_CLAUDE_06_release.md',
       all_45_terminal=all_terminal,all_45_usable=all_usable,
       smallcell_12_gate=gate(small12,12),smallcell_11_same_protocol_sensitivity=gate(small,11),maincell_16_gate=gate(large,16),
       smallcell_protocol_note='S00reusedFalse/e6;11newAuto/e5; keepboth12mixedand11sameprotocolresults visible',
       LREAL_paired_check=pair_gate,numerical_convergence=convergence,crosssite=cross,relaxation_cost=cost,
       CPU_actual_core_hours=budget['cumulative_actual_core_hours'],CPU_remaining_authorized_core_hours=3000-budget['cumulative_actual_core_hours'],
       GPU_actual_card_hours=gb['actual_card_hours'],GPU_remaining_authorized_card_hours=40-gb['actual_card_hours'],
       training_released=False)
    save(C/'CLAUDE06_SCIENCE.json',result)
    table=['| 主机 | 作业 | 状态 | 实耗 | 电子/离子检查 |','|---|---|---|---:|---|']
    for host,rows in [('CPU',cpu),('GPU',gpu)]:
        for r in rows:
            check='通过' if r['usable'] else '待完成' if r['state'] not in ['COMPLETED','FAILED','TIMEOUT','CANCELLED'] else '未通过/不可作标签'
            table.append(f'| {host} | {r["name"]} / {r["slurm_job_id"]} | {r["state"]} | {r["actual"]:.4f} | {check} |')
    def gate_text(g):
        if g['status']!='CHECKED':return f'待数据：有效{g["n_valid"]}/{g["n_required"]}'
        return f'{"通过" if g["passed"] else "未通过"}；Spearman={g["spearman"]}，MACE前三留在DFT前六={g["top3_retained_in_DFT_top6"]}/3，力RMSE中位数={g["force_component_RMSE_median_eV_A"]:.5f}eV/Å，逐原子误差P95={g["atom_force_error_p95_eV_A"]:.5f}eV/Å'
    cross_text='\n'.join(f'- {v["name"]}（{"理想格点起始" if v["from_ideal"] else "MACE几何起始"}）：DFT末态占位类别{"改变" if v["DFT"]["occupation_class_changed"] else "未改变"}；与MACE末态占位类别{"一致" if v["same_MACE_DFT_occupation_class"] else "不同"}；最大格点匹配距离{v["DFT"]["max_assignment_distance_A"]:.3f}Å。' for v in cross)
    control=load(C/'REVIEW_STATUS.json',{})
    halt_text=''
    if control.get('claude06_halt_cause_job_id'):
        halt_text='''## 停批状态

S08/k333（1886892）在1740秒内部限时触发后退出，父作业FAILED/exit124，共1741秒、15.475556核时；未完成电子收敛，末步能量不得作DFT标签。其余19个CPU作业正常完成。按既定失败规则，取消了尚未开始的GPU598/600/602/604/606/608，均未消耗卡时；这些是策略取消，不是六项VASP物理失败。停批时已运行的S02/592与L00/596按原限额执行并回收证据，最新状态见下表；没有自动重试。P2的S08/k333及六项已取消工作未完成，整轮校准和训练不放行。诊断见[停批记录](CLAUDE06_HALT_20261023.md)。

'''
    text=f'''# Claude06校准结果（持续更新）

更新UTC：{result['updated_at_utc']}。授权依据：[审核06](REVIEW_CLAUDE_06_release.md)。具体设置、限额及登记门槛见[执行登记](CLAUDE06_EXECUTION.md)。**本轮不放行LLM训练。**

本批45项全部终态：{all_terminal}；全部电子/离子检查通过：{all_usable}。作业结束不等于MLIP门槛通过，门槛另见下文。

{halt_text}

## MLIP门槛

- 小胞12项：{gate_text(result['smallcell_12_gate'])}。
- 小胞同协议11项敏感性：{gate_text(result['smallcell_11_same_protocol_sensitivity'])}。
- 主胞16项：{gate_text(result['maincell_16_gate'])}。

S00复用原False/EDIFF1e-6；新11项为Auto/1e-5。12项结果包含此设置差异，附11项同协议结果。三组LREAL配对完成{len(pair)}/3；不以绝对能共同偏移判失败。大胞Γ到k222检查本轮未放行。

主胞P2b原始XML、运行时输入指纹、排序和力误差的独立复核如已完成，见[复核记录](calibration/evidence/Claude06_P2b_independent_verification.json)。报告仅对本校准池作门槛判断。

## 跨位和完整弛豫

已取得电子和离子均收敛的弛豫末态{len(cross)}/9，其中理想格点起始{sum(r['from_ideal'] for r in cross)}/4。详细初态/MACE/DFT格点匹配、空位和NN/NNN结果见[结构诊断JSON](calibration/CLAUDE06_SCIENCE.json)。静态优化跨位不代表有限温度扩散率或离子电导。未收敛输出仅作进度诊断。

{cross_text}

## 分池预算

CPU累计{result['CPU_actual_core_hours']:.6f}/3000核时，授权余额{result['CPU_remaining_authorized_core_hours']:.6f}核时；含剩余限额的总预留{budget['actual_plus_remaining_caps']:.6f}核时。

GPU累计{result['GPU_actual_card_hours']:.6f}/40卡时，授权余额{result['GPU_remaining_authorized_card_hours']:.6f}卡时；含剩余限额的总预留{gb['actual_plus_remaining_caps']:.6f}卡时。余额是项目授权余额，不是平台账户实时余额。所有失败与启动诊断均计入。

## 原始作业

'''+ '\n'.join(table)+'\n'
    final_audit=ROOT/'CLAUDE06_FINAL_AUDIT.md'
    if all_terminal and final_audit.exists():
        text+='\n\n'+final_audit.read_text()
    (ROOT/'CLAUDE06_RESULTS.md').write_text(text)
    (ROOT/'BUDGET_REPLAN.md').write_text(f'''# Claude06预算更新

CPU池3000核时：历史253.68，本轮20×16=320，提交前上限573.68；当前实耗{result['CPU_actual_core_hours']:.6f}，未使用授权{result['CPU_remaining_authorized_core_hours']:.6f}。

GPU池40卡时：历史0.074166667，P2b16×0.25=4，9项弛豫各239分钟合计35.85，提交前总上限39.924166667；当前实耗{result['GPU_actual_card_hours']:.6f}，未使用授权{result['GPU_remaining_authorized_card_hours']:.6f}。

完整清单与执行口径见[CLAUDE06_EXECUTION.md](CLAUDE06_EXECUTION.md)。历史2888核时清单已被逐轮重排，不能再与本轮重复相加；平台账户余额需另行核验。未授权训练或生产任务。

剩余已提交作业限额：CPU {budget['remaining_claude06_core_hour_caps']:.6f}核时，GPU {gb['remaining_card_hour_caps']:.6f}卡时。若本批均已终态，缺失或已取消工作须经人工复核后另行放行，不按剩余额度自动补算。弛豫成本及缺项见[回收审核](CLAUDE06_FINAL_AUDIT.md)。
''')
    print(json.dumps(dict(small12=result['smallcell_12_gate'],small11=result['smallcell_11_same_protocol_sensitivity'],main16=result['maincell_16_gate'],converged_relaxations=len(cross)),indent=2))

if __name__=='__main__':main()
