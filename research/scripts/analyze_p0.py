"""Parse retrieved VASP results and accounting; no compute submission or inference."""
from pathlib import Path
import csv,datetime,json,re,xml.etree.ElementTree as ET
import numpy as np
from ase.io import read
from claude03_gate import p1_gate
from review05_gate import relative_gate
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration'
ledger=json.loads((C/'P0_SUBMISSIONS.json').read_text())
if (C/'P1_SUBMISSIONS.json').exists():ledger['jobs']+=json.loads((C/'P1_SUBMISSIONS.json').read_text())['jobs']
if (C/'FAST_SUBMISSIONS.json').exists():ledger['jobs']+=json.loads((C/'FAST_SUBMISSIONS.json').read_text())['jobs']
if (C/'REFERENCE_SUBMISSIONS.json').exists():ledger['jobs']+=json.loads((C/'REFERENCE_SUBMISSIONS.json').read_text())['jobs']
snapshot=json.loads((C/'P0_LATEST.json').read_text())
lines=snapshot['sacct']['stdout'].splitlines();head=lines[0].split('|')
accounting=[dict(zip(head,l.split('|'))) for l in lines[1:]]
parents={r['JobIDRaw']:r for r in accounting if '.' not in r['JobIDRaw']}
results=[]
for attempt in ledger['jobs']:
    jid=attempt['slurm_job_id'];d=C/'P0_outputs'/attempt['name']
    result=dict(attempt,accounting=parents.get(jid,{}),step_accounting=[r for r in accounting if r['JobIDRaw'].startswith(jid+'.')])
    if (d/'vasprun.xml').exists() and parents.get(jid,{}).get('State')=='COMPLETED':
        xml=ET.parse(d/'vasprun.xml').getroot();calcs=xml.findall('calculation');calc=calcs[-1]
        def param(name):
            node=xml.find('.//parameters//i[@name="'+name+'"]')
            return node.text.strip() if node is not None else None
        energies={i.attrib['name']:float(i.text) for i in calc.findall('energy/i')}
        forces=np.array([[float(x) for x in v.text.split()] for v in calc.findall('varray[@name="forces"]/v')])
        positions=np.array([[float(x) for x in v.text.split()] for v in calc.findall('structure/varray[@name="positions"]/v')])
        out=(d/'OUTCAR').read_text(errors='replace')
        result['vasp']={'energy_eV':energies,'n_scf_steps':len(calc.findall('scstep')),
                        'nelm':int(param('NELM')),'nbands':int(param('NBANDS')),
                        'n_kpoints':len(xml.findall('kpoints/varray[@name="kpointlist"]/v')),
                        'lreal':param('LREAL'),'ediff':float(param('EDIFF')),'n_atoms':len(forces),'n_ionic_outputs':len(calcs),
                        'EDIFF_reached_marker':'aborting loop because EDIFF is reached' in out,
                        'max_force_eV_A':float(np.linalg.norm(forces,axis=1).max()),
                        'forces_eV_A':forces.tolist(),'fractional_positions':positions.tolist()}
        for name,pattern in [('elapsed_seconds',r'Elapsed time \(sec\):\s*([\d.]+)'),
                             ('total_cpu_seconds',r'Total CPU time used \(sec\):\s*([\d.]+)'),
                             ('maximum_memory_kB_rank0',r'Maximum memory used \(kb\):\s*([\d.]+)')]:
            m=re.search(pattern,out,re.I)
            if m:result['vasp'][name]=float(m.group(1))
        loops=re.findall(r'LOOP:\s+cpu time\s+([\d.]+):\s+real time\s+([\d.]+)',out)
        result['vasp']['SCF_loop_real_seconds']=[float(t) for _,t in loops]
        result['vasp']['SCF_loop_real_seconds_mean']=float(np.mean([float(t) for _,t in loops])) if loops else None
        result['vasp']['SCF_loop_real_seconds_median']=float(np.median([float(t) for _,t in loops])) if loops else None
        result['vasp']['SCF_converged']=result['vasp']['EDIFF_reached_marker'] and len(calc.findall('scstep'))<int(param('NELM'))
        logical_name=attempt.get('logical_job',attempt['name'])
        config=attempt.get('config') or next(j['config'] for j in json.loads((C/'jobs.json').read_text()) if j['id']==logical_name)
        cached=read(C/'geometries'/f'{config}_relaxed.extxyz')
        if cached.calc is not None and 'forces' in cached.calc.results:
            fractional_delta=positions-cached.get_scaled_positions(wrap=False)
            fractional_delta-=np.rint(fractional_delta)
            assert np.max(np.linalg.norm(researcher@example.org,axis=1))<1e-6
            xml_symbols=[rc.find('c').text.strip() for rc in xml.findall('atominfo/array[@name="atoms"]/set/rc')]
            assert xml_symbols==cached.get_chemical_symbols()
            delta=forces-cached.calc.results['forces']
            result['vasp']['DFT_vs_cached_MACE_force_component_RMSE_eV_A']=float(np.sqrt(np.mean(delta**2)))
            result['vasp']['DFT_vs_cached_MACE_max_atom_force_error_eV_A']=float(np.linalg.norm(delta,axis=1).max())
    elif (d/'OSZICAR').exists():
        printed=re.findall(r'^\s*(?:DAV|RMM|CG):\s*(\d+)\s+', (d/'OSZICAR').read_text(errors='replace'),re.M)
        if printed:result['partial_vasp']={'last_printed_scf_step':int(printed[-1]),'SCF_converged':False,'usable_energy_label':False}
        if printed and (d/'OUTCAR').exists():
            partial_out=(d/'OUTCAR').read_text(errors='replace')
            times=[float(t) for t in re.findall(r'LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)',partial_out)]
            result['partial_vasp'].update(SCF_loop_real_seconds=times,
                SCF_loop_real_seconds_mean=float(np.mean(times)) if times else None,
                SCF_loop_real_seconds_median=float(np.median(times)) if times else None)
        stdout=d/('slurm-'+jid+'.out')
        if stdout.exists():
            result['runtime_diagnostics']={'BRMIX_serious_count':stdout.read_text(errors='replace').count('BRMIX: very serious problems')}
    results.append(result)
logical={}
for r in results:logical[r.get('logical_job',r['name'])]=r
comparisons={}
for first,second,label in [('01_S00_sp16','02_S00_sp32','16_vs_32'),
                           ('03_L00_gamma','03b_L00_gamma_lrealauto','LREAL_False_vs_Auto'),
                           ('04_L01_gamma','fast_L01','L01_old_vs_fast'),
                           ('ref_L00','fast_L00','L00_ref_vs_fast')]:
    a=logical.get(first,{});b=logical.get(second,{})
    if a.get('vasp',{}).get('SCF_converged') and b.get('vasp',{}).get('SCF_converged'):
        assert np.allclose(a['vasp']['fractional_positions'],b['vasp']['fractional_positions'],atol=1e-8)
        fa=np.array(a['vasp']['forces_eV_A']);fb=np.array(b['vasp']['forces_eV_A']);diff=fb-fa
        ea=a['vasp']['energy_eV'];eb=b['vasp']['energy_eV']
        comparisons[label]={'first_attempt':a['name'],'second_attempt':b['name'],
          'delta_E0_eV':eb['e_0_energy']-ea['e_0_energy'],
          'delta_TOTEN_eV':eb['e_fr_energy']-ea['e_fr_energy'],
          'force_component_RMSE_eV_A':float(np.sqrt(np.mean(diff**2))),
          'max_atom_force_difference_eV_A':float(np.linalg.norm(diff,axis=1).max()),
          'Slurm_wall_speedup_first_over_second':float(a['accounting']['ElapsedRaw'])/float(b['accounting']['ElapsedRaw']),
          'core_hour_ratio_second_over_first':float(b['accounting']['CPUTimeRAW'])/float(a['accounting']['CPUTimeRAW'])}
report={'analyzed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'attempts':results,'comparisons':comparisons,
        'P0_all_five_converged':len(logical)==5 and all(r.get('vasp',{}).get('SCF_converged',False) for r in logical.values())}
status=json.loads((C/'REVIEW_STATUS.json').read_text())
required=status.get('required_p0_completed_points',list(logical))
report['P0_reviewed_required_points_converged']=all(logical.get(name,{}).get('vasp',{}).get('SCF_converged',False) for name in required)
report['P1_all_eleven_converged']=all(logical.get(f'{i+4:02d}_S{i:02d}_sp16',{}).get('vasp',{}).get('SCF_converged',False) for i in range(1,12))
if all(name in logical and 'vasp' in logical[name] for name in ['03_L00_gamma','04_L01_gamma']):
    left=logical['03_L00_gamma']['vasp'];right=logical['04_L01_gamma']['vasp']
    assert left['lreal']==right['lreal']=='F'
    report['comparisons']['L01_minus_L00_LREAL_False']={
        'delta_E0_eV':right['energy_eV']['e_0_energy']-left['energy_eV']['e_0_energy'],
        'delta_TOTEN_eV':right['energy_eV']['e_fr_energy']-left['energy_eV']['e_fr_energy']}
if all(logical.get(name,{}).get('vasp',{}).get('SCF_converged') for name in ['fast_L00','fast_L01']):
    left=logical['fast_L00']['vasp'];right=logical['fast_L01']['vasp']
    assert left['lreal']==right['lreal'] and left['ediff']==right['ediff']==1e-5
    report['comparisons']['L01_minus_L00_fast']={
        'delta_E0_eV':right['energy_eV']['e_0_energy']-left['energy_eV']['e_0_energy'],
        'delta_TOTEN_eV':right['energy_eV']['e_fr_energy']-left['energy_eV']['e_fr_energy'],
        'protocol':'Claude04 fast:gam/Auto/1e-5/AMIX0.2/BMIX0.0001'}
report['fast_both_converged']=all(logical.get(name,{}).get('vasp',{}).get('SCF_converged') for name in ['fast_L00','fast_L01'])
if (C/'CLAUDE05_JOB_LIST.json').exists():
    gate={'passed':False,'status':'AWAITING_REFERENCE','energy_tolerance_eV_cell':.02,'max_atom_force_tolerance_eV_A':.02,'n_configurations':2,'approval_reference':'REVIEW_CLAUDE_05_fast_gpu.md'}
    required=['ref_L00','04_L01_gamma','fast_L00','fast_L01']
    if all(logical.get(name,{}).get('vasp',{}).get('SCF_converged') for name in required):
        r00=logical['ref_L00']['vasp'];r01=logical['04_L01_gamma']['vasp']
        assert r00['lreal']==r01['lreal']=='F' and r00['ediff']==r01['ediff']==1e-6
        force_error=max(report['comparisons'][label]['max_atom_force_difference_eV_A'] for label in ['L01_old_vs_fast','L00_ref_vs_fast'])
        gate.update(relative_gate(r00['energy_eV']['e_0_energy'],r01['energy_eV']['e_0_energy'],
                    logical['fast_L00']['vasp']['energy_eV']['e_0_energy'],
                    logical['fast_L01']['vasp']['energy_eV']['e_0_energy'],force_error),status='CHECKED')
    report['Claude05_relative_energy_gate']=gate
if (C/'CLAUDE03_JOB_LIST.json').exists():
    report['P1_release_gate']=p1_gate(report,json.loads((C/'BUDGET.json').read_text()))
    (C/'P1_GATE.json').write_text(json.dumps(dict(report['P1_release_gate'],checked_at_utc=report['analyzed_at_utc']),indent=2)+'\n')
(C/'P0_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
with (C/'SCF_TIMINGS.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['logical_job','attempt','slurm_job_id','SCF_step_1based','wall_seconds_rank0'])
    for r in results:
        for i,t in enumerate(r.get('vasp',r.get('partial_vasp',{})).get('SCF_loop_real_seconds',[]),1):
            writer.writerow([r.get('logical_job',r['name']),r['name'],r['slurm_job_id'],i,t])
print(json.dumps({'P0_all_five_converged':report['P0_all_five_converged'],
 'results':[{'name':r['name'],'state':r['accounting'].get('State'),'vasp':{k:v for k,v in r.get('vasp',{}).items() if k not in ['forces_eV_A','fractional_positions','SCF_loop_real_seconds']}} for r in results],
 'comparisons':comparisons},indent=2))
