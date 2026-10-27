"""General fluorite occupations and a bounded, review-only VASP job design."""
from functools import lru_cache
import numpy as np
import spglib
from ase.spacegroup import crystal


def reference(n):
    return crystal(['Zr','O'],basis=[(0,0,0),(.25,.25,.25)],spacegroup=225,
                   cellpar=[5.14,5.14,5.14,90,90,90]).repeat((n,n,n))


def make_state(seed,n,n_y):
    s=reference(n);rng=np.random.default_rng(seed)
    cats=np.flatnonzero(s.numbers==40);oxy=np.flatnonzero(s.numbers==8)
    if n_y%2 or not 0<n_y<len(cats):raise ValueError('Invalid charge-compensated count')
    s.numbers[rng.choice(cats,n_y,replace=False)]=39
    s.numbers[rng.choice(oxy,n_y//2,replace=False)]=0
    return s


@lru_cache(maxsize=2)
def symmetry_permutations(n):
    s=reference(n);f=s.get_scaled_positions();g=4*n
    grid=np.rint(f*g).astype(int)%g
    lookup=np.full((g,g,g),-1,dtype=int);lookup[tuple(grid.T)]=np.arange(len(s))
    sym=spglib.get_symmetry((s.cell.array,f,s.numbers),symprec=1e-5)
    out=[]
    for rot,t in zip(sym['rotations'],sym['translations']):
        moved=((f@rot.T+t)*g)%g
        indices=np.rint(moved).astype(int)%g
        perm=lookup[tuple(indices.T)]
        if np.any(perm<0) or len(np.unique(perm))!=len(s):raise ValueError('Symmetry not bijective')
        out.append(perm)
    return np.unique(out,axis=0)


def canonical_key(state,n):
    inv=np.argsort(symmetry_permutations(n),axis=1)
    return min(row.astype(np.uint8).tobytes() for row in state.numbers[inv]).hex()


def shell_counts(state):
    p=reference(round((len(state)/12)**(1/3)))
    dist=p.get_all_distances(mic=True)
    cats=np.flatnonzero(p.numbers==40);oxy=np.flatnonzero(p.numbers==8)
    shells=np.unique(np.round(dist[np.ix_(cats,oxy)],6))
    y=np.flatnonzero(state.numbers==39);vac=np.flatnonzero(state.numbers==0)
    yv=dist[np.ix_(y,vac)]
    return {'Y_vac_NN':int(np.sum(np.isclose(yv,shells[0],atol=1e-5))),
            'Y_vac_NNN':int(np.sum(np.isclose(yv,shells[1],atol=1e-5)))}


def job_specs():
    rows=[]
    def add(cid,kind,phase,cores,hours,k=(2,2,2),encut=520,variant='base',lo=.1,hi=.5):
        jid=f'{len(rows)+1:02d}_{cid}_{variant}'
        rows.append(dict(id=jid,config=cid,kind=kind,phase=phase,cores=cores,
                         limit_hours=hours,kmesh=list(k),encut=encut,variant=variant,
                         estimated_wall_hours_low=lo,estimated_wall_hours_high=hi,
                         hard_core_hours=cores*hours,
                         estimated_core_hours_low=cores*lo,estimated_core_hours_high=cores*hi))
    add('S00','sp','P0_cost',16,.5,variant='sp16',lo=1/12,hi=.5)
    add('S00','sp','P0_cost',32,.5,variant='sp32',lo=.05,hi=.5)
    for cid in ['L00','L01']:
        add(cid,'sp','P0_cost',32,2,k=(1,1,1),variant='gamma',lo=.25,hi=2)
    for i in range(1,12):
        add(f'S{i:02d}','sp','P1_small',16,1,variant='sp16',lo=1/12,hi=.5)
    for cid in ['S00','S04','S08']:
        add(cid,'sp','P2_convergence',16,2,encut=650,variant='encut650',lo=.25,hi=1.5)
        add(cid,'sp','P2_convergence',16,2,k=(3,3,3),variant='k333',lo=.25,hi=1.5)
    for cid in ['L00','L01']:
        add(cid,'sp','P2_maincell',32,8,variant='k222',lo=1,hi=8)
    for cid in ['S00','S01','S04','S05','S08','S09']:
        add(cid,'relax','P4_small_relax',16,16,variant='relax',lo=.5,hi=8)
    add('L00','pilot_relax','P3_mainpilot',32,8,variant='pilot8',lo=.5,hi=4)
    # Preserve all reviewed IDs while removing two superseded full relaxations.
    rows=[j for j in rows if j['id'] not in ['27_S05_relax','29_S09_relax']]
    for j in rows:
        j.update(lreal='.FALSE.',input_geometry='frozen_MACE_CPU_float64_relaxed')
    auto=next(j.copy() for j in rows if j['id']=='03_L00_gamma')
    auto.update(id='03b_L00_gamma_lrealauto',variant='gamma_lrealauto',lreal='Auto')
    rows.insert(3,auto)
    for num,cid in [(31,'S02'),(32,'S06')]:
        rows.append(dict(id=f'{num}_{cid}_relax_fromideal',config=cid,kind='relax',
                         phase='P4_crosssite',cores=16,limit_hours=16,kmesh=[2,2,2],
                         encut=520,variant='relax_fromideal',lreal='.FALSE.',
                         input_geometry='ideal_initial_occupation',
                         estimated_wall_hours_low=.5,estimated_wall_hours_high=8,
                         hard_core_hours=256,estimated_core_hours_low=8,
                         estimated_core_hours_high=128))
    return rows
