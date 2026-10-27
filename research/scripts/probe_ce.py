"""icet two-sublattice CE + canonical MC plumbing with frozen-MLIP labels only."""
import csv, json, time
from pathlib import Path
import numpy as np
import torch
from ase.io import read,write
from icet import ClusterSpace, ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble
from sklearn.linear_model import Ridge
from mace.calculators import MACECalculator
from lattice import reference_lattice, physical_atoms
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
t0=time.perf_counter();torch.set_num_threads(4)
states=read(OUT/'random_100_lattice.extxyz',index=':')
with (OUT/'random_energies_cpu.csv').open() as f: rows=list(csv.DictReader(f))
energies=np.array([float(x['energy_eV']) for x in rows])
# Both sublattices must remain explicit; X is vacancy only for CE.
parent=reference_lattice()
allowed=[['Zr','Y'] if z==40 else ['O','X'] for z in parent.numbers]
cs=ClusterSpace(parent,cutoffs=[5.2,3.0],chemical_symbols=allowed)
X=np.array([cs.get_cluster_vector(s) for s in states])
train=np.arange(80);test=np.arange(80,100)
center=float(energies[train].mean());target=(energies-center)/len(parent)
reg=Ridge(alpha=1e-6,fit_intercept=False).fit(X[train],target[train])
ce=ClusterExpansion(cs,reg.coef_)
ce.write(OUT/'smoke.ce')
calc=ClusterExpansionCalculator(states[0],ce)
mc=CanonicalEnsemble(states[0],calc,temperature=1000,random_seed=20261022,
                     dc_filename=str(OUT/'ce_mc.dc'),ensemble_data_write_interval=100,
                     trajectory_write_interval=100)
initial=float(calc.calculate_total(occupations=states[0].numbers))
tmc=time.perf_counter();mc.run(1000);mc_time=time.perf_counter()-tmc
final=mc.structure
write(OUT/'ce_mc_final_lattice.extxyz',final)
assert sorted(final.numbers)==sorted(states[0].numbers)
atoms=physical_atoms(final)
atoms.calc=MACECalculator(model_paths=str(ROOT/'models/mace-small-density-agnesi-stress.model'),
                           device='cpu',default_dtype='float32')
final_ml=float(atoms.get_potential_energy())
pred=(X@reg.coef_)*len(parent)+center
result={'label_source':'MACE-MP-0b2 small; unrelaxed, not DFT',
        'n_states':len(states),'n_train':80,'n_test':20,'n_cluster_functions':len(cs),
        'cutoffs_A':[5.2,3.0],'ridge_alpha':1e-6,'fit_data_matrix_rank':int(np.linalg.matrix_rank(X[train])),
        'rmse_train_eV_per_cell':float(np.sqrt(np.mean((pred[train]-energies[train])**2))),
        'rmse_test_eV_per_cell':float(np.sqrt(np.mean((pred[test]-energies[test])**2))),
        'mc_trials':1000,'mc_seconds':mc_time,'temperature_K':1000,
        'initial_ce_relative_total_eV':initial,
        'final_ce_relative_total_eV':float(ce.predict(final)*len(parent)),
        'final_mlip_unrelaxed_total_eV':final_ml,'initial_mlip_total_eV':float(energies[0]),
        'composition_preserved':True,'elapsed_seconds':time.perf_counter()-t0,
        'interpretation':'Feasibility only; neither CE accuracy nor ground state validated.'}
(OUT/'probe_ce.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
