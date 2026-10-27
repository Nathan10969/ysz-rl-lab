"""Audit element order, periodic site drift, symmetry duplication, and input hashes."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from ase.io import read
from ase.geometry import find_mic
from calibration import reference,canonical_key,shell_counts
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'calibration'
out=[]
for rec in json.loads((P/'configurations.json').read_text()):
 cid=rec['id'];n=2 if rec['n_sites']==96 else 3
 parent=reference(n);initial=read(P/'geometries'/f'{cid}_initial.extxyz')
 atoms=read(P/'geometries'/f'{cid}_relaxed.extxyz');ids=atoms.arrays['site_id']
 disp=find_mic(atoms.positions-parent.positions[ids],parent.cell,pbc=True)[1]
 mapped=parent.copy();mapped.numbers[:]=0;assigned=np.full(len(atoms),-1,dtype=int);assignment_distance=np.full(len(atoms),np.nan)
 for mask,sites in [(np.isin(atoms.numbers,[39,40]),np.flatnonzero(parent.numbers==40)),
                    (atoms.numbers==8,np.flatnonzero(parent.numbers==8))]:
  ai=np.flatnonzero(mask)
  vec=atoms.positions[ai,None,:]-parent.positions[sites][None,:,:]
  dist=find_mic(vec.reshape(-1,3),parent.cell,pbc=True)[1].reshape(len(ai),len(sites))
  ii,jj=linear_sum_assignment(dist**2);assigned[ai[ii]]=sites[jj];assignment_distance[ai[ii]]=dist[ii,jj]
 assert np.all(assigned>=0) and len(set(assigned))==len(atoms)
 mapped.numbers[assigned]=atoms.numbers
 rows={'id':cid,'canonical_key_initial':canonical_key(initial,n),'canonical_key_relaxed_site_mapping':canonical_key(mapped,n),
       'occupation_class_changed':canonical_key(initial,n)!=canonical_key(mapped,n),
       'n_atoms_mapped_to_different_sites':int(np.sum(assigned!=ids)),
       'max_displacement_from_initial_site_A':float(disp.max()),'max_final_site_assignment_distance_A':float(assignment_distance.max()),
       'NN_NNN_before':shell_counts(initial),'NN_NNN_after_mapping':shell_counts(mapped)}
 out.append(rows)
(P/'geometry_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'n_configs':len(out),'n_changed_classes':sum(x['occupation_class_changed'] for x in out),'max_displacement_A':max(x['max_displacement_from_initial_site_A'] for x in out)},indent=2))
