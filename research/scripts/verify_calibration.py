"""Validate review inputs without running VASP, SSH, sbatch, or MLIP."""
from pathlib import Path
import csv,datetime,hashlib,json,re,subprocess
import numpy as np
from ase.io import read
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'calibration'
jobs=json.loads((P/'jobs.json').read_text());budget=json.loads((P/'BUDGET.json').read_text());configs=json.loads((P/'configurations.json').read_text());audit=json.loads((P/'geometry_audit.json').read_text())
checks={};errors=[]
checks['31_unique_review_jobs']=len(jobs)==31 and len({x['id'] for x in jobs})==31
checks['14_prepared_configs']=len(configs)==14 and all(x['converged'] for x in configs)
checks['allocation_caps_and_smoke_fit_3000']=sum(x['cores']*x['limit_hours'] for x in jobs)+16*23/3600<=3000
checks['budget_arithmetic_matches']=sum(x['hard_core_hours'] for x in jobs)==budget['hard_core_hours']==2888
checks['first_wave_216']=sum(x['hard_core_hours'] for x in jobs if x['phase']=='P0_cost')==216
checks['legacy_review_inputs_only_16_or_32_cores']=all(x['cores'] in [16,32] for x in jobs)
checks['only_P0_approved']=all(x['review_status']==('APPROVED_P0' if x['phase']=='P0_cost' else 'HELD_FOR_POST_P0_REVIEW') for x in jobs)
seen_geometry={}
for j in jobs:
 d=P/'01_calib'/j['id'];a=read(d/'POSCAR',format='vasp')
 n=j['n_atoms'];expected=[28,4,62] if n==94 else [92,16,208]
 symbols=a.get_chemical_symbols()
 assert symbols==['Zr']*expected[0]+['Y']*expected[1]+['O']*expected[2],j['id']
 assert len(a)==n and abs(4*expected[0]+3*expected[1]-2*expected[2])==0
 target=read(P/'geometries'/f'{j["config"]}_{"initial" if j["input_geometry"]=="ideal_initial_occupation" else "relaxed"}.extxyz')
 if j['input_geometry']=='ideal_initial_occupation':
  target=target[target.numbers!=0]
  target=target[sorted(range(len(target)),key=lambda i:{'Zr':0,'Y':1,'O':2}[target[i].symbol])]
 assert np.allclose(a.positions,target.positions,atol=1e-9)
 assert np.allclose(a.cell.array,target.cell.array,atol=1e-9)
 geometry_hash=hashlib.sha256((d/'POSCAR').read_bytes()).hexdigest()
 geometry_key=(j['config'],j['input_geometry'])
 if geometry_key in seen_geometry:assert seen_geometry[geometry_key]==geometry_hash
 seen_geometry[geometry_key]=geometry_hash
 incar={}
 for line in (d/'INCAR').read_text().splitlines():
  if '=' in line:
   k,v=line.split('=',1);incar[k.strip()]=v.strip()
 assert int(incar['ENCUT'])==j['encut']
 assert int(incar['NBANDS'])==(480 if n==94 else 1536)
 assert int(incar['NBANDS'])>(12*expected[0]+11*expected[1]+6*expected[2])/2
 assert j['cores']%int(incar['KPAR'])==0
 assert (j['cores']//int(incar['KPAR']))%int(incar['NCORE'])==0
 expect_nsw=200 if j['kind']=='relax' else 8 if j['kind']=='pilot_relax' else 0
 assert int(incar['NSW'])==expect_nsw
 assert int(incar['ISYM'])==0
 assert incar['LREAL']==j['lreal']
 if expect_nsw:assert float(incar['EDIFFG'])<0 and int(incar['ISIF'])==2
 assert list(map(int,(d/'KPOINTS').read_text().splitlines()[3].split()))==j['kmesh']
 script=(d/'sub.sh').read_text();seconds=round(j['limit_hours']*3600)
 time_tag=f'{seconds//3600:02d}:{seconds%3600//60:02d}:00'
 assert f'#SBATCH --time={time_tag}' in script
 assert f'#SBATCH --ntasks={j["cores"]}' in script
 assert '--nodes=1' in script and '--no-requeue' in script and '--mem-per-cpu=1500M' in script
 assert 'sbatch ' not in script and 'srun ' not in script
 res=subprocess.run(['/bin/bash','-n',str(d/'sub.sh')],capture_output=True,text=True)
 assert res.returncode==0,(j['id'],res.stderr)
 for name,want in j['input_hashes'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==want
 mapping=json.loads((d/'atom_mapping.json').read_text())
 assert mapping['counts']==expected and mapping['order']==['Zr','Y','O']
 assert len(set(mapping['vasp_atom_index_0based_to_parent_site']))==n
checks['all_poscars_formula_order_same_geometry']=True
checks['all_incar_kpoints_parallel_and_walltimes_match']=True
checks['all_31_batch_scripts_parse_and_hashes_match']=True
by_id={j['id']:j for j in jobs}
false_incar=(P/'01_calib/03_L00_gamma/INCAR').read_text()
auto_incar=(P/'01_calib/03b_L00_gamma_lrealauto/INCAR').read_text()
def settings(text):return {k.strip():v.strip() for line in text.splitlines() if '=' in line for k,v in [line.split('=',1)] if k.strip()!='SYSTEM'}
assert settings(auto_incar)==dict(settings(false_incar),LREAL='Auto')
checks['Auto_only_physical_setting_difference']=True
checks['only_four_MACE_initialized_full_relaxations']=len([j for j in jobs if j['phase']=='P4_small_relax'])==4
checks['two_ideal_initialized_crosssite_controls']=sum(j['input_geometry']=='ideal_initial_occupation' for j in jobs)==2
checks['all_atom_order_mappings_match_POTCAR_order']=True
checks['no_POTCAR_in_inputs']=not any(x.name=='POTCAR' for x in (P/'01_calib').rglob('*'))
checks['12_unique_small_relaxed_occupations']=len({x['canonical_key_relaxed_site_mapping'] for x in audit if x['id'].startswith('S')})==12
checks['2_unique_large_relaxed_occupations']=len({x['canonical_key_relaxed_site_mapping'] for x in audit if x['id'].startswith('L')})==2
checks['cross_site_changes_recorded']=sum(x['occupation_class_changed'] for x in audit)==10
recovery_dirs={'01p_S00_sp16':'01_S00_sp16','02p_S00_sp32':'02_S00_sp32',
               '03p_L00_gamma':'03_L00_gamma','03bp_L00_gamma_lrealauto':'03b_L00_gamma_lrealauto',
               '04p_L01_gamma':'04_L01_gamma'}
for name,logical in recovery_dirs.items():
 d=P/'recovery_inputs'/name;meta=json.loads((d/'recovery.json').read_text())
 for input_name in ['POSCAR','INCAR','KPOINTS']:
  assert (d/input_name).read_bytes()==(P/'01_calib'/logical/input_name).read_bytes()
 for input_name,want in meta['input_hashes'].items():assert hashlib.sha256((d/input_name).read_bytes()).hexdigest()==want
 script=(d/'sub.sh').read_text()
 assert f'--mem={meta["cores"]*1500}M' in script and 'srun --mpi=pmi2' in script
 assert 'I_MPI_PMI_LIBRARY=/opt/hpc/slurm/21.08.6/lib/libpmi2.so' in script
 assert '--cpu-bind=cores' in script and '--no-requeue' in script
 assert subprocess.run(['/bin/bash','-n',str(d/'sub.sh')]).returncode==0
checks['five_runtime_corrected_inputs_match_reviewed_physics_and_hashes']=True
checks['P0_used_plus_remaining_caps_within_216']=budget['p0_actual_plus_remaining_caps']<=216
checks['used_plus_all_remaining_caps_within_3000']=budget['actual_plus_remaining_caps']<=3000
if (P/'CLAUDE03_JOB_LIST.json').exists():
 current=json.loads((P/'CLAUDE03_JOB_LIST.json').read_text())
 assert len(current)==14
 for j in current:
  d=P/'execution_claude03'/j['name'];old=P/'01_calib'/j['logical_job']
  assert j['cores']==(64 if j['phase']=='P0_cost' else 32)
  assert j['limit_hours']==(1 if j['phase']=='P0_cost' else .5)
  assert j['hard_core_hours']==(64 if j['phase']=='P0_cost' else 16)
  for input_name in ['INCAR','KPOINTS','POSCAR']:assert (d/input_name).read_bytes()==(old/input_name).read_bytes()
  for input_name,want in j['input_hashes'].items():assert hashlib.sha256((d/input_name).read_bytes()).hexdigest()==want
  script=(d/'sub.sh').read_text()
  assert f'#SBATCH --ntasks={j["cores"]}' in script and f'#SBATCH --mem={j["cores"]*1500}M' in script
  assert 'srun --mpi=pmi2' in script and '--no-requeue' in script and 'SCF not converged; stop dependency chain' in script
  assert subprocess.run(['/bin/bash','-n',str(d/'sub.sh')]).returncode==0
  assert j['nbands']%((j['cores']//j['kpar'])//j['ncore'])==0
 checks['Claude03_3x64_11x32_inputs_physics_and_runtime_validated']=True
if (P/'CLAUDE04_JOB_LIST.json').exists():
 current=json.loads((P/'CLAUDE04_JOB_LIST.json').read_text())
 assert [j['name'] for j in current]==['05f_L00','06f_L01']
 assert sum(j['hard_core_hours'] for j in current)==256
 for j in current:
  d=P/'execution_claude04'/j['name'];old=P/'execution_claude03'/j['physical_source_attempt']
  assert j['cores']==128 and j['limit_hours']==1 and j['binary']=='vasp_gam'
  for name in ['POSCAR','KPOINTS','atom_mapping.json']:assert (d/name).read_bytes()==(old/name).read_bytes()
  before,after=settings((old/'INCAR').read_text()),settings((d/'INCAR').read_text())
  assert after==dict(before,EDIFF='1E-5',LREAL='Auto',AMIX='0.2',BMIX='0.0001')
  assert list(map(int,(d/'KPOINTS').read_text().splitlines()[3].split()))==[1,1,1]
  for name,want in j['input_hashes'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==want
  script=(d/'sub.sh').read_text()
  assert '--ntasks=128' in script and '--mem=192000M' in script and '--time=01:00:00' in script
  assert '/vasp_gam"' in script and '/vasp_std"' not in script
  assert 'srun --mpi=pmi2' in script and '--cpu-bind=cores' in script and '--no-requeue' in script
  assert 'sha256sum -c REVIEWED_INPUTS.sha256' in script and j['binary_sha256'] in script
  assert subprocess.run(['/bin/bash','-n',str(d/'sub.sh')]).returncode==0
  assert j['nbands']%((j['cores']//j['kpar'])//j['ncore'])==0
 status=json.loads((P/'REVIEW_STATUS.json').read_text())
 assert status['approval_reference'] in ['REVIEW_CLAUDE_04_L00.md','REVIEW_CLAUDE_05_fast_gpu.md']
 assert status['p1_state'] in ['HELD_CLAUDE04','HELD_CLAUDE05']
 assert json.loads((P/'P1_GATE.json').read_text())['allowed'] is False
 assert not (P/'P1_SUBMISSIONS.json').exists()
 assert budget['fast_reallocated_from_unreleased_p2plus_core_hours']==256
 assert budget['remaining_p2plus_core_hour_caps']==(1728 if budget.get('cpu_future_scope')=='94_ATOM_ONLY_AFTER_GPU_SELECTION' else 2496-256)
 checks['Claude04_two_fast_128core_same_geometries_only_approved_changes']=True
 checks['Claude04_P1_held_and_fast_budget_reallocated']=True
if (P/'CLAUDE05_JOB_LIST.json').exists():
 current=json.loads((P/'CLAUDE05_JOB_LIST.json').read_text());assert len(current)==1
 j=current[0];d=P/'execution_claude05'/j['name'];old=P/'execution_claude03/03q_L00_gamma64'
 assert j['cores']==128 and j['limit_hours']==1.5 and j['hard_core_hours']==192
 assert settings((d/'INCAR').read_text())==dict(settings((old/'INCAR').read_text()),AMIX='0.2',BMIX='0.0001')
 for name in ['POSCAR','KPOINTS','atom_mapping.json']:assert (d/name).read_bytes()==(old/name).read_bytes()
 for name,want in j['input_hashes'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==want
 script=(d/'sub.sh').read_text()
 assert '--ntasks=128' in script and '--time=01:30:00' in script and '--mem=192000M' in script
 assert '/vasp_std"' in script and 'srun --mpi=pmi2' in script and '--no-requeue' in script
 assert '5300s srun' in script and subprocess.run(['/bin/bash','-n',str(d/'sub.sh')]).returncode==0
 assert json.loads((P/'P1_GATE.json').read_text())['allowed'] is False
 checks['Claude05_ref_same_geometry_and_only_approved_mixing_changes']=True
for file in ['PLAN.md','CALIBRATION_REVIEW.md']:
 for target in re.findall(r'\]\(([^)]+)\)',(ROOT/file).read_text()):
  if not target.startswith(('http:','https:','#')) and not (ROOT/target).exists():errors.append(f'{file}: {target}')
checks['all_local_report_links_resolve']=not errors
manifest=[]
for p in sorted((P/'01_calib').rglob('*')):
 if p.is_file():manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for name in recovery_dirs:
 for p in sorted((P/'recovery_inputs'/name).iterdir()):
  if p.is_file():manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
if (P/'execution_claude03').exists():
 for p in sorted((P/'execution_claude03').rglob('*')):
  if p.is_file():manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
if (P/'execution_claude04').exists():
 for p in sorted((P/'execution_claude04').rglob('*')):
  if p.is_file():manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
if (P/'execution_claude05').exists():
 for p in sorted((P/'execution_claude05').rglob('*')):
  if p.is_file():manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for name in ['PLAN.md','CALIBRATION_REVIEW.md','calibration/JOB_LIST.csv','calibration/CONFIGURATIONS.csv','calibration/configurations.json','calibration/geometry_audit.json','calibration/BUDGET.json','calibration/maincell_timing.json']:
 p=ROOT/name;manifest.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(P/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
report={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_passed':all(checks.values()),'checks':checks,'errors':errors,'new_jobs_submitted':json.loads((P/'REVIEW_STATUS.json').read_text())['new_cloud_jobs_submitted'],'hard_core_hours':budget['hard_core_hours'],'reserved_after_smoke_and_caps':budget['reserve_after_all_time_limits'],'inputs_hashed':len(manifest)}
(P/'VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert all(checks.values())
