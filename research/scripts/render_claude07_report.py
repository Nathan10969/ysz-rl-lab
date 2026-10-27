"""Render Review07 scientific gates from recovered local evidence only."""
import argparse
import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read
from scipy.optimize import linear_sum_assignment

from calibration import canonical_key, reference, shell_counts
from claude07_safety import TERMINAL, atomic_json, budget_status, last_complete_calculation
from collect_claude07 import POOLS, geometry_matches, load

ROOT = Path(__file__).resolve().parents[1]


def energy(row):
    return row['energy_eV']['e_0_energy']


def g2_gate(rows, gamma):
    expected = ['B10', 'B11', 'B15', 'B00']
    good = {r['config']: r for r in rows if r.get('usable') and r.get('input_geometry_matches_mace')}
    result = dict(status='AWAITING_DATA', required=expected,
                  available=sorted(set(good)&set(expected)&set(gamma)),
                  supports_SP_plus_topk_relaxation=None, top3_order_changed=None,
                  criterion='span(relaxed_minus_Gamma_SP)<0.1 eV/cell AND all exact occupations unchanged',
                  geometry='DFT relaxation starts from the same MACE geometry as Claude06 P2b Gamma SP')
    if not all(k in good and k in gamma and good[k].get('occupancy') for k in expected):
        return result
    changes = {k: energy(good[k])-gamma[k] for k in expected}
    anchor = changes['B10']
    unchanged = all(good[k]['occupancy'].get('same_exact_occupation',
                    good[k]['occupancy'].get('same_MACE_DFT_occupation_class')) for k in expected)
    span = max(changes.values())-min(changes.values())
    before = sorted(expected, key=lambda k: gamma[k])[:3]
    after = sorted(expected, key=lambda k: energy(good[k]))[:3]
    original_three = ['B10', 'B11', 'B15']
    result.update(status='CHECKED', relative_reference='B10',
                  relative_correction_span_eV=span, exact_occupations_unchanged=bool(unchanged),
                  supports_SP_plus_topk_relaxation=bool(span < .1 and unchanged),
                  Gamma_SP_top3_among_four=before, relaxed_top3_among_four=after,
                  top3_order_changed=before != after,
                  original_B10_B11_B15_order_changed=sorted(original_three, key=lambda k: gamma[k]) !=
                                                      sorted(original_three, key=lambda k: energy(good[k])),
                  recommended_final_protocol='MACE_geometry_DFT_SP_plus_topk_full_relaxation_checks'
                    if span < .1 and unchanged else 'topk_full_DFT_relaxation_required',
                  per_configuration=[dict(config=k, Gamma_SP_eV=gamma[k], relaxed_eV=energy(good[k]),
                      relaxed_minus_SP_eV=changes[k], relative_relax_minus_SP_eV=changes[k]-anchor,
                      SP_relative_to_B10_eV=gamma[k]-gamma['B10'],
                      relaxed_relative_to_B10_eV=energy(good[k])-energy(good['B10']),
                      actual_card_hours=good[k]['actual'],
                      n_ionic_outputs=good[k].get('convergence', {}).get('n_ionic_outputs'),
                      occupancy=good[k]['occupancy']) for k in expected])
    return result


def g3_gate(rows, gamma):
    expected = ['B10', 'B07', 'B00']
    good = {r['config']: r for r in rows if r.get('usable') and r.get('input_geometry_matches_mace')}
    result = dict(status='AWAITING_DATA', required=expected, passed=None,
                  available=sorted(set(good)&set(gamma)&set(expected)),
                  criterion='max(pairwise relative-energy change)<=0.05 eV/cell',
                  geometry='Both Gamma and k222 are single points on the same MACE geometries; no G2 final geometry')
    if not all(k in good and k in gamma for k in expected):
        return result
    shifts = {k: energy(good[k])-gamma[k] for k in expected}
    pairs = [dict(config_i=i, config_j=j, corrected_relative_change_eV=shifts[i]-shifts[j])
             for i, j in combinations(expected, 2)]
    span = max(shifts.values())-min(shifts.values())
    result.update(status='CHECKED', passed=bool(span <= .05),
                  max_pairwise_relative_change_eV=span, pairs=pairs,
                  per_configuration=[dict(config=k, Gamma_SP_eV=gamma[k], k222_SP_eV=energy(good[k]),
                                          k222_minus_Gamma_eV=shifts[k],
                                          actual_card_hours=good[k]['actual']) for k in expected])
    return result


def corrected_convergence(old, s08_k333_delta=None, protocol_verified=False):
    offsets = [r['Auto_minus_False_eV'] for r in old.get('LREAL_paired_check', {}).get('rows', [])]
    offset = float(np.mean(offsets)) if offsets else None
    rows = []
    for item in old.get('numerical_convergence', []):
        raw = dict(item['per_config_high_minus_base_eV'])
        if item['variant'] == 'k333' and s08_k333_delta is not None:
            raw['S08'] = s08_k333_delta
        corrected = dict(raw)
        if 'S00' in corrected and offset is not None:
            corrected['S00'] -= offset
        span = max(corrected.values())-min(corrected.values()) if len(corrected) >= 2 else None
        rows.append(dict(variant=item['variant'], n_configs=len(corrected),
                         per_config_raw_high_minus_base_eV=raw,
                         per_config_corrected_high_minus_base_eV=corrected,
                         Auto_minus_False_common_offset_eV=offset,
                         S00_evidence_kind='common_offset_inference',
                         S00_direct_same_protocol_measurement=False,
                         protocol_verified=protocol_verified,
                         relative_energy_change_span_eV=span,
                         within_0_02_eV=bool(span <= .02) if span is not None and offset is not None and protocol_verified else None,
                         three_point_update_complete={'S00', 'S04', 'S08'} <= set(corrected),
                         note='S00 is corrected by the three-pair mean Auto-False offset; it is not a directly measured Auto/e5 S00 baseline.'))
    return rows


def occupancy(atoms, parent, initial):
    """Map every atom, using its full geometry, onto the complete mother lattice."""
    if not np.allclose(atoms.cell.array, parent.cell.array, atol=1e-5):
        raise ValueError('Mother lattice and relaxed cell differ; affine mapping requires separate review')
    mapped = parent.copy()
    mapped.numbers[:] = 0
    assigned = np.full(len(atoms), -1, dtype=int)
    distances = np.zeros(len(atoms))
    for mask, sites in [(np.isin(atoms.numbers, [39, 40]), np.flatnonzero(parent.numbers == 40)),
                        (atoms.numbers == 8, np.flatnonzero(parent.numbers == 8))]:
        ai = np.flatnonzero(mask)
        vectors = atoms.positions[ai, None, :]-parent.positions[sites][None, :, :]
        distance = find_mic(vectors.reshape(-1, 3), parent.cell, pbc=True)[1].reshape(len(ai), len(sites))
        ii, jj = linear_sum_assignment(distance**2)
        assigned[ai[ii]] = sites[jj]
        distances[ai[ii]] = distance[ii, jj]
    if np.any(assigned < 0) or len(set(assigned)) != len(atoms):
        raise ValueError('Incomplete or nonunique mother-lattice assignment')
    mapped.numbers[assigned] = atoms.numbers
    n = round((len(parent)/12)**(1/3))
    return dict(canonical_key=canonical_key(mapped, n),
                exact_occupation_numbers=mapped.numbers.tolist(),
                occupation_class_changed=canonical_key(mapped, n) != canonical_key(initial, n),
                vacancy_site_ids=np.flatnonzero((mapped.numbers == 0)&(parent.numbers == 8)).tolist(),
                Y_site_ids=np.flatnonzero(mapped.numbers == 39).tolist(),
                assigned_site_ids=assigned.tolist(), max_assignment_distance_A=float(distances.max()),
                shell_counts=shell_counts(mapped), evidence='full_geometry_plus_complete_mother_lattice')


def mapped_comparison(root, job, atoms):
    initial = read(root/job['parent_geometry'])
    n = round((len(initial)/12)**(1/3))
    parent = reference(n)
    initial_source = 'complete_mother_lattice_including_vacancy_X_sites'
    if len(parent) != len(initial):
        sites = initial.arrays.get('site_id')
        if (sites is None or sites.shape != (len(initial),) or
                not np.issubdtype(sites.dtype, np.integer) or
                np.any(sites < 0) or np.any(sites >= len(parent)) or
                len(set(sites)) != len(initial)):
            raise ValueError('Physical parent geometry requires unique integer mother-lattice site_id values')
        if (not np.all(initial.pbc) or
                not np.allclose(initial.cell.array, parent.cell.array, atol=1e-5, rtol=0)):
            raise ValueError('Physical parent geometry must use the periodic mother-lattice cell')
        distances = find_mic(initial.positions-parent.positions[sites], parent.cell, pbc=True)[1]
        if np.any(distances > 1e-5):
            raise ValueError('Physical parent geometry positions do not match their mother-lattice site_id values')
        cations = parent.numbers[sites] == 40
        if (not np.all(np.isin(initial.numbers[cations], [39, 40])) or
                not np.all(initial.numbers[~cations] == 8) or
                not set(np.flatnonzero(parent.numbers == 40)) <= set(sites)):
            raise ValueError('Physical parent geometry species or vacancies violate the mother sublattices')
        complete = parent.copy()
        complete.numbers[:] = 0
        complete.numbers[sites] = initial.numbers
        initial = complete
        initial_source = 'physical_atoms_reconstructed_from_verified_ideal_site_id_values'
    mace = read(root/job['mace_geometry'])
    dft_map = occupancy(atoms, parent, initial)
    mace_map = occupancy(mace, parent, initial)
    exact = dft_map['exact_occupation_numbers'] == mace_map['exact_occupation_numbers']
    canonical = dft_map['canonical_key'] == mace_map['canonical_key']
    return dict(DFT=dft_map, MACE=mace_map, same_exact_occupation=exact,
                same_MACE_DFT_occupation_class=canonical,
                initial_occupancy_source=initial_source,
                symmetry_equivalent_but_exact_sites_differ=bool(canonical and not exact))


def atoms_from_geometry(geometry):
    return Atoms(symbols=geometry['symbols'], cell=geometry['cell_A'],
                 scaled_positions=geometry['scaled_positions'], pbc=geometry.get('pbc', True))


def verify_protocol(root):
    evidence = []
    names = ['c06_S00_encut650', 'c06_S04_encut650', 'c06_S08_encut650',
             'c06_S00_k333', 'c06_S04_k333']
    for name in names:
        path = root/'calibration/execution_claude06'/name/'INCAR'
        text = path.read_text() if path.exists() else ''
        lreal = re.search(r'^\s*LREAL\s*=\s*(\S+)', text, re.M)
        ediff = re.search(r'^\s*EDIFF\s*=\s*(\S+)', text, re.M)
        runtime_path = root/'calibration/claude06_outputs'/name/'runtime_input_hashes.sha256'
        runtime_text = runtime_path.read_text() if runtime_path.exists() else ''
        runtime_hash = re.search(r'^([0-9a-f]{64})\s+INCAR\s*$', runtime_text, re.M)
        hash_matches = bool(path.exists() and runtime_hash and hashlib.sha256(path.read_bytes()).hexdigest() == runtime_hash.group(1))
        okay = bool(lreal and lreal.group(1).lower() == 'auto' and ediff and float(ediff.group(1)) == 1e-5 and hash_matches)
        evidence.append(dict(path=str(path.relative_to(root)), LREAL=lreal.group(1) if lreal else None,
                             EDIFF=ediff.group(1) if ediff else None, verified=okay,
                             runtime_INCAR_hash_matches=hash_matches,
                             runtime_hash_path=str(runtime_path.relative_to(root))))
    return dict(verified=all(r['verified'] for r in evidence), rows=evidence,
                source='Claude06 INCAR values checked against recovered runtime INCAR SHA256 fingerprints')


def gamma_references(root, old_gpu):
    energies = {}
    checks = []
    for config in ['B10', 'B11', 'B15', 'B07', 'B00']:
        row = next((r for r in old_gpu if r.get('config') == config and r.get('phase') == 'P2b_maincell' and r.get('usable')), None)
        if not row:
            continue
        path = root/'calibration_gpu/claude06_outputs'/row['name']/'vasprun.xml'
        try:
            frame = last_complete_calculation(path.read_text())
            matches = geometry_matches(frame, root/'calibration/claude06_pool'/f'{config}_relaxed.extxyz')
            if matches:
                energies[config] = energy(row)
            checks.append(dict(config=config, path=str(path.relative_to(root)), same_MACE_geometry=matches))
        except Exception as error:
            checks.append(dict(config=config, error=repr(error)))
    return energies, checks


def g1_report(root, rows, manifest):
    result = dict(status='AWAITING_DATA', source_frame_1based=108,
                  original_card_hours=3.9672222222222224, original_complete_ionic_outputs=108,
                  total_ionic_outputs=None, total_card_hours=None,
                  same_final_occupation_as_MACE=None,
                  cost_interpretation='Ideal-start L00 plus continuation; this is not a MACE-start relaxation cost.')
    job = next((r for r in manifest if r.get('group') == 'G1'), None)
    if not job:
        return result
    source = root/'calibration_gpu/claude06_outputs/c06_L00_relax_fromideal/vasprun.xml'
    try:
        frame = last_complete_calculation(source.read_text(), expected_index=108)
        geometry = dict(symbols=frame['symbols'], cell_A=frame['basis'], scaled_positions=frame['positions'])
        result['frame108_diagnostic'] = mapped_comparison(root, job, atoms_from_geometry(geometry))
        result['frame108_fmax_eV_A'] = float(np.linalg.norm(frame['forces'], axis=1).max())
        result['source_has_incomplete_trailing_step'] = frame['trailing_xml_incomplete']
    except Exception as error:
        result['frame108_diagnostic_error'] = repr(error)
    row = next((r for r in rows if r.get('group') == 'G1'), None)
    if not row:
        return result
    result.update(continuation_state=row['state'], continuation_card_hours=row['actual'],
                  total_card_hours=result['original_card_hours']+row['actual'])
    outputs = row.get('last_complete_frame', {}).get('completed_calculations')
    if outputs is not None:
        result.update(continuation_complete_ionic_outputs=outputs, total_ionic_outputs=108+outputs,
                      output_count_note='108 + continuation complete outputs; the continuation first evaluates the restarted frame108 coordinates, so this sum is not a count of unique geometries.')
    if row.get('usable') and row.get('occupancy'):
        result.update(status='CHECKED', final_occupancy=row['occupancy'],
                      same_final_occupation_as_MACE=row['occupancy']['same_MACE_DFT_occupation_class'],
                      same_exact_final_occupation_as_MACE=row['occupancy']['same_exact_occupation'],
                      final_ionic_converged=row['convergence']['ionic_converged'])
    return result


def render(root, smoke_home=None):
    cpu_home = root/POOLS['CPU']['home']
    gpu_home = root/POOLS['DFT_GPU']['home']
    cpu = load(cpu_home/'ANALYSIS.json', {'jobs': []})['jobs']
    gpu = load(gpu_home/'ANALYSIS.json', {'jobs': []})['jobs']
    manifest = load(gpu_home/'JOB_LIST.json', [])
    for row in gpu:
        if row.get('final_geometry'):
            try:
                row['occupancy'] = mapped_comparison(root, row, atoms_from_geometry(row['final_geometry']))
            except Exception as error:
                row['occupancy_error'] = repr(error)
    old = load(root/'calibration/CLAUDE06_SCIENCE.json', {})
    old_gpu = load(root/'calibration_gpu/CLAUDE06_ANALYSIS.json', {'jobs': []})['jobs']
    old_cpu = load(root/'calibration/CLAUDE06_ANALYSIS.json', {'jobs': []})['jobs']
    gamma, reference_checks = gamma_references(root, old_gpu)
    s08 = next((r for r in cpu if r.get('group') == 'C1' and r.get('usable') and r.get('input_geometry_matches_mace')), None)
    base = next((r for r in old_cpu if r.get('config') == 'S08' and r.get('phase') == 'P1_small' and r.get('usable')), None)
    delta = energy(s08)-energy(base) if s08 and base else None
    protocol = verify_protocol(root)
    budgets = {}
    for name, settings in POOLS.items():
        jobs = load(root/settings['home']/'JOB_LIST.json', [])
        fallback = budget_status(jobs, {}, settings['historical'], settings['pool_limit'], settings['round_limit'])
        fallback.update(pool=name, unit=settings['unit'], review07_actual=0,
                        accounting_status='NOT_COLLECTED; frozen Claude06 baseline plus prepared reservation')
        budgets[name] = load(root/settings['home']/'BUDGET.json', fallback)
    smoke_home = Path(smoke_home) if smoke_home else root/'training_smoke'
    budgets['TRAINING_SMOKE'] = load(smoke_home/'BUDGET.json',
        dict(pool='TRAINING_SMOKE', pool_limit=10, unit='card_hours', accounting_status='NOT_COLLECTED',
             separate_from_DFT_pool=True, formal_training_released=False))
    smoke_analysis = load(smoke_home/'ANALYSIS.json', dict(jobs=[], all_jobs_terminal=False))
    review08_authorization = load(smoke_home/'REVIEW08_AUTHORIZATION.json', {})
    gpu_limit = load(root/'calibration/REVIEW_STATUS.json', {}).get('max_concurrent_gpu', 2)
    if type(gpu_limit) is not int or gpu_limit not in (1, 2):
        raise ValueError('Review07 GPU concurrency must be 1 or 2')
    result = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  approval_reference='REVIEW_CLAUDE_07_gates.md',
                  G1=g1_report(root, gpu, manifest),
                  G2=g2_gate([r for r in gpu if r.get('group') == 'G2'], gamma),
                  G3=g3_gate([r for r in gpu if r.get('group') == 'G3'], gamma),
                  recovered_DFT_geometry_diagnostics=[{k:r.get(k) for k in
                      ['name', 'group', 'config', 'state', 'usable', 'occupancy', 'occupancy_error']}
                      for r in gpu if r.get('final_geometry')],
                  C1=dict(state=s08['state'] if s08 else next((r['state'] for r in cpu if r.get('group') == 'C1'), 'NOT_SUBMITTED'),
                          S08_k333_high_minus_base_eV=delta,
                          numerical_convergence=corrected_convergence(old, delta, protocol['verified'])),
                  convergence_protocol_evidence=protocol, Gamma_reference_geometry_checks=reference_checks,
                  budgets=budgets, max_concurrent_gpu=gpu_limit,
                  scheduling_amendment=load(smoke_home/'SCHEDULE_AMENDMENTS.json', {}),
                  active_priority_amendment=load(smoke_home/'SINGLE_GPU_PRIORITY_AMENDMENT.json', {}),
                  smoke_workload_amendment=load(smoke_home/'WORKLOAD_AMENDMENTS.json', {}),
                  training_smoke=smoke_analysis,
                  review08_smoke_authorization=review08_authorization,
                  all_DFT_jobs_terminal=len(cpu)==1 and len(gpu)==8 and all(r['state'] in TERMINAL for r in cpu+gpu),
                  formal_training_released=False,
                  smoke_authorization='Independent 10 card-hour pool, <=2 card-hours/job, tiny Qwen3-0.6B SFT and 316-atom MACE throughput only.',
                  training_policy='No formal training, multi-round RSI, or full comparative experiments released by this report.')
    atomic_json(cpu_home/'SCIENCE.json', result)
    lines = [
        '# Claude07 校准结果（持续更新）', '',
        '更新 UTC：'+result['updated_at_utc']+'。授权依据：[审核07](REVIEW_CLAUDE_07_gates.md)。', '',
        '**正式训练未放行。** 训练冒烟池独立封顶 10 卡时，条件GRPO授权见后文。'+
        ('用户要求单卡串行，为师弟保留另一张卡。' if gpu_limit == 1 else '平台按两卡并发估算。'), '',
        '## G1：L00 从第 108 个完整帧续算', '',
        '状态：'+result['G1']['status']+'。旧段 108 个完整离子输出、3.967222 卡时；续算与旧段成本相加，属于理想格点起始的总成本。',
    ]
    first = result['G1']
    if 'frame108_diagnostic' in first:
        lines.append('免费第 108 帧诊断：与 MACE 占位类别一致 = '+str(first['frame108_diagnostic']['same_MACE_DFT_occupation_class'])+
                     '；格点逐位一致 = '+str(first['frame108_diagnostic']['same_exact_occupation'])+'。')
    if first['total_card_hours'] is not None:
        if first['status'] == 'CHECKED':
            lines.append(f"两段总成本 {first['total_card_hours']:.6f} 卡时；总完整离子输出 {first['total_ionic_outputs']}；收敛末态与 MACE 占位类别一致 = {first['same_final_occupation_as_MACE']}。")
        else:
            lines.append(f"两段截至本次查询累计实耗 {first['total_card_hours']:.6f} 卡时；续算状态 {first.get('continuation_state', '待查询')}，最终离子步数与收敛占位待回收核验。")
        lines.append('续算首步重算第 108 帧坐标，因此两段输出数之和不等于独立几何数。')
    lines += ['', '## G2：终评协议', '',
              '状态：'+result['G2']['status']+'。四个构型在同一 MACE 几何单点基准上比较：先求各自弛豫修正，再减去 B10 修正；跨度严格小于 0.1 eV/胞且所有逐位占位不变，才支持“DFT 单点 + top-k 完整弛豫抽检”。']
    second = result['G2']
    if second['status'] == 'CHECKED':
        lines += [f"修正跨度 {second['relative_correction_span_eV']:.6f} eV/胞；支持单点加抽检 = {second['supports_SP_plus_topk_relaxation']}；原 B10/B11/B15 顺序改变 = {second['original_B10_B11_B15_order_changed']}。",
                  '弛豫前/后四项中的前三：'+str(second['Gamma_SP_top3_among_four'])+' → '+str(second['relaxed_top3_among_four'])+'。', '',
                  '| 构型 | 卡时 | 完整离子输出 | 相对能修正 eV/胞 | 逐位占位相同 | 对称类别相同 |',
                  '|---|---:|---:|---:|---|---|']
        for row in second['per_configuration']:
            occ = row['occupancy']
            lines.append(f"| {row['config']} | {row['actual_card_hours']:.6f} | {row['n_ionic_outputs']} | {row['relative_relax_minus_SP_eV']:.6f} | {occ.get('same_exact_occupation')} | {occ['same_MACE_DFT_occupation_class']} |")
        lines.append('若对称类别相同但逐位格点不同，JSON 另标为对称等价，不记为占位完全未变。')
    else:
        for row in gpu:
            if row.get('group') == 'G2' and row.get('usable') and row.get('occupancy'):
                lines.append(f"已回收 {row['config']}：{row['actual']:.6f} 卡时、{row['convergence']['n_ionic_outputs']} 个完整离子输出；"
                             f"与起始 MACE 逐位占位一致 = {row['occupancy']['same_exact_occupation']}，"
                             f"对称类别一致 = {row['occupancy']['same_MACE_DFT_occupation_class']}。四项齐全后再判定修正跨度和前三顺序。")
    lines += ['', '## G3：大胞 k 点', '',
              '状态：'+result['G3']['status']+'。B10/B07/B00 的 k222 与 Γ 均用各自相同 MACE 几何。逐对比较能量差的改变量，最大值须 ≤0.05 eV/胞；不使用 G2 弛豫末态。']
    third = result['G3']
    if third['status'] == 'CHECKED':
        lines.append(f"最大逐对相对能改变量 {third['max_pairwise_relative_change_eV']:.6f} eV/胞；通过 = {third['passed']}。")
    lines += ['', '## C1：S08/k333 与原数值收敛协议', '',
              '高精度变体 Auto/EDIFF=1E-5 及运行时 INCAR 指纹核验：'+str(protocol['verified'])+'。S00 按三组 Auto−False 均值修正共同偏移，明确属于推断，不是直接测量了同协议 S00。', '']
    for row in result['C1']['numerical_convergence']:
        span = row['relative_energy_change_span_eV']
        lines.append(f"- {row['variant']}：{row['n_configs']} 点；修正跨度 {span:.6f} eV/胞；≤0.02 判据 = {row['within_0_02_eV']}；三点齐全 = {row['three_point_update_complete']}。" if span is not None else '- '+row['variant']+'：待数据。')
    lines += ['', '## 分池记账', '',
              '现行费用以新一轮 [CPU 预算](calibration/claude07/BUDGET.json)、[DFT GPU 预算](calibration_gpu/claude07/BUDGET.json) 和 [训练冒烟预算](training_smoke/BUDGET.json) 为准。旧 Claude06 账本只作为冻结历史基数。', '',
              '| 池 | 累计实耗 | 含剩余封顶预留 | 池额度 |', '|---|---:|---:|---:|']
    for name, budget in budgets.items():
        actual = budget.get('actual', budget.get('actual_card_hours', budget.get('actual_gpu_hours', '未回收')))
        planned = budget.get('actual_plus_remaining_caps',
                             budget.get('planned_hard_gpu_hours', '未回收'))
        authorized = budget.get('pool_limit', budget.get('authorized_card_hours', budget.get('authorized_gpu_hours', '?')))
        lines.append('| '+name+' | '+str(actual)+' | '+str(planned)+' | '+str(authorized)+' |')
    lines += ['', 'CPU 仅计父作业 CPUTimeRAW；GPU 仅计父作业 ElapsedRaw × 实际分配卡数。步骤账目只作诊断，不重复收费。失败和启动诊断均计入，各池不得互借。', '',
              '## 作业状态', '', '| 池 | 作业 / ID | 状态 | 本次实耗 | 可作标签 |', '|---|---|---|---:|---|']
    for name, rows in [('CPU', cpu), ('DFT_GPU', gpu)]:
        for row in rows:
            lines.append(f"| {name} | {row['name']} / {row['slurm_job_id']} | {row['state']} | {row['actual']:.6f} | {row['usable']} |")
    lines += ['', '## 两项训练冒烟', '',
              ('当前已提交 SFT 633（≤0.50卡时）与 MACE 635（≤0.75卡时）。按最新用户优先级，'
               'B11/621结束后依次635→629→625→633；G2结果不是训练科学闸门。'
               if result['active_priority_amendment'] else
               '当前原清单已提交 SFT 633（≤0.50卡时）与 MACE 635（≤0.75卡时），资源依赖见调度记录。'), '',
              '| 作业 / ID | 状态 | 测量记录可用 |', '|---|---|---|']
    for row in smoke_analysis.get('jobs', []):
        lines.append(f"| {row.get('id')} / {row.get('slurm_job_id')} | {row.get('state')} | {row.get('usable', False)} |")
    if review08_authorization:
        grpo = review08_authorization.get('conditional_grpo', {})
        default_reservation = 0 if grpo.get('state', '').startswith('NOT_RELEASED') else grpo.get('cap_card_hours', 0)
        extra_reservation = float(review08_authorization.get('additional_conditional_reservation_card_hours', default_reservation))
        current_projection = float(budgets['TRAINING_SMOKE'].get('actual_plus_remaining_caps', 0)) + extra_reservation
        lines += ['', f"审核08新增一次条件GRPO冒烟，封顶 {grpo.get('cap_card_hours')} 卡时；状态 {grpo.get('state')}，job ID={grpo.get('slurm_job_id')}。",
                  f"仍属同一个10卡时池，当前已用加剩余作业/条件预留合计 {current_projection:.6f} 卡时；未安排 {10-current_projection:.6f} 卡时。",
                  '上表训练池账本仅统计原633/635，审核08条件预留另列于 [授权记录](training_smoke/REVIEW08_AUTHORIZATION.json)。须原两项回收核验且v2合法率≥95%才进入GRPO更新，正式训练未批准。']
        if grpo.get('state', '').startswith('NOT_RELEASED'):
            lines += [f"当前不释放GRPO：{grpo.get('not_released_reason', grpo['state'])}。原2卡时条件预留未转为作业，不自动重试或追加任务。"]
    else:
        lines += ['', '独立10卡时池其余8.75未分配，不自动使用。']
    lines += ['', '训练步数、token、保存/重载、batch等价性、吞吐与内存的原始指标见 [冒烟分析](training_smoke/ANALYSIS.json) 和 outputs/。只有成功回收并核验的测量才能替换训练计划的待测项；短样本和40步有界FIRE不充当2048-token训练或完整BFGS成本。', '',
              '详细几何、占位、力、SCF 时间与原始账目见新一轮 [SCIENCE.json](calibration/claude07/SCIENCE.json)、各池 ANALYSIS.json 和 outputs/。未完成数据不产生正面科学结论。', '']
    (root/'CLAUDE07_RESULTS.md').write_text('\n'.join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--smoke-home', type=Path, help='Independent smoke directory containing BUDGET.json')
    args = parser.parse_args(argv)
    result = render(args.root, args.smoke_home)
    print(json.dumps(dict(G1=result['G1']['status'], G2=result['G2']['status'], G3=result['G3']['status'],
                          C1=result['C1']['state'], formal_training_released=False), indent=2))


if __name__ == '__main__':
    main()
