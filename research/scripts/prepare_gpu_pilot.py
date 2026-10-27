"""Prepare the explicitly authorized 1/2-A100 L01 comparisons; no calculations."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'
REMOTE='/CLUSTER_HOME/runs/01_calib_gpu'
GAM_SHA='deb79c3ffcef4faa1e05df3d9c704ff532ac9de9cdde581a6911a5b2ee3ee326'
POT_SHA={'Zr_sv':'25aed69cb10325f9d37c5c68912b61a17387d1f8e4f1d804860ffa10c8a4bf76',
         'Y_sv':'5e7f7496c6fa99024fec326bcd60dcee8ff7e886f9613d31da35c685a538a036',
         'O':'818f92134a0a090dccd8ba1447fa70422a3b330e708bb4f08108d8ae51209ddf'}

def main():
    assert not G.exists(),'GPU inputs already exist: inspect and reconcile; do not overwrite'
    G.mkdir();(G/'evidence').mkdir();jobs=[]
    source=C/'execution_claude04/06f_L01'
    srcmeta=json.loads((source/'job.json').read_text())
    for filename,digest in srcmeta['input_hashes'].items():
        assert hashlib.sha256((source/filename).read_bytes()).hexdigest()==digest
    for name,ngpu in [('07g_L01_1g',1),('08g_L01_2g',2)]:
        d=G/'inputs'/name;d.mkdir(parents=True)
        for filename in ['POSCAR','KPOINTS','atom_mapping.json']:shutil.copy2(source/filename,d/filename)
        old=(source/'INCAR').read_text()
        assert 'NCORE = 4' in old and 'KPAR = 1' in old and 'NSIM' not in old
        (d/'INCAR').write_text(old.replace('NCORE = 4','NCORE = 1'))
        # One MPI rank per GPU; six physical CPU cores per rank, matching platform template.
        batch=f'''#!/bin/bash
# User-authorized A100 pilot only; no requeue/retries or later-phase submissions.
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks={ngpu}
#SBATCH --ntasks-per-node={ngpu}
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:{ngpu}
#SBATCH --mem={90*ngpu}G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=ysz_{name}
#SBATCH --chdir={REMOTE}/{name}
#SBATCH --output=slurm-%j.out
set -eu
# End before Slurm's 30-second KillWait could exceed the total three-card-hour cap.
exec timeout --signal=TERM --kill-after=10s 3540s bash run.sh
'''
        runner='''#!/bin/bash
set -eo pipefail
ulimit -s unlimited
ulimit -l unlimited
source /etc/profile
module purge
module load mkl/latest nvhpc/25.3
set -u
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export MKL_NUM_THREADS=6
sha256sum -c REVIEWED_INPUTS.sha256
VASP=/data/software/app/vasp.6.5.1/bin/vasp_gam
P="$HOME/Jony/software/potpaw_vasp/PBE"
'''
        runner+=f"printf '%s  %s\\n' '{GAM_SHA}' \"$VASP\" | sha256sum -c -\n"
        for specie,digest in POT_SHA.items():
            runner+=f"printf '%s  %s\\n' '{digest}' \"$P/{specie}/POTCAR\" | sha256sum -c -\n"
        runner+='''cat "$P/Zr_sv/POTCAR" "$P/Y_sv/POTCAR" "$P/O/POTCAR" > POTCAR
sha256sum INCAR KPOINTS POSCAR POTCAR > runtime_input_hashes.sha256
date -Is
scontrol show job "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\\n' "${CUDA_VISIBLE_DEVICES:-unset}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv -l 10 > gpu_samples.csv &
SAMPLE_PID=$!
trap 'kill "$SAMPLE_PID" 2>/dev/null || true; wait "$SAMPLE_PID" 2>/dev/null || true' EXIT
mpirun -np "$SLURM_NTASKS" --map-by slot:PE=6 --bind-to core --report-bindings "$VASP" > log.txt 2>&1
python3 - <<'SCF_CHECK'
from pathlib import Path
out=Path('OUTCAR').read_text(errors='replace')
if 'aborting loop because EDIFF is reached' not in out:
    raise SystemExit('SCF not converged: stop dependent pilot, no retries')
print('SCF completion marker confirmed')
SCF_CHECK
date -Is
'''
        (d/'sub.sh').write_text(batch);(d/'run.sh').write_text(runner)
        reviewed={f:hashlib.sha256((d/f).read_bytes()).hexdigest() for f in ['POSCAR','INCAR','KPOINTS','atom_mapping.json','run.sh']}
        (d/'REVIEWED_INPUTS.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in reviewed.items()))
        for filename in ['sub.sh','run.sh']:
            (d/filename).chmod(0o755)
            subprocess.run(['/bin/bash','-n',str(d/filename)],check=True)
        hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.iterdir()}
        job={'name':name,'config':'L01','gpus':ngpu,'mpi_ranks':ngpu,'cpus_per_task':6,
             'host_cpus':6*ngpu,'memory_GiB':90*ngpu,'limit_hours':1,'hard_card_hours':ngpu,
             'internal_deadline_seconds':3540,'binary':'vasp_gam','binary_sha256':GAM_SHA,
             'remote_directory':REMOTE+'/'+name,'reference_cpu_job_id':'1882636',
             'physical_source':'calibration/execution_claude04/06f_L01',
             'only_INCAR_difference_from_CPU':{'NCORE':[4,1]},'kpar':1,'NSIM':'VASP default',
             'POTCAR_order':['Zr_sv','Y_sv','O'],'POTCAR_hashes':POT_SHA,'input_hashes':hashes}
        (d/'job.json').write_text(json.dumps(job,indent=2)+'\n');jobs.append(job)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (G/'JOB_LIST.json').write_text(json.dumps(jobs,indent=2)+'\n')
    budget={'authorization':'User directly relayed A100 supplement: two L01 pilots, total at most3card-hours',
            'authorized_card_hours':3,'hard_card_hours':3,'actual_card_hours':0,'remaining_card_hour_caps':3,
            'actual_plus_remaining_caps':3,'known_prior_project_gpu_card_hours':0,
            'whole_account_balance_snapshot_card_hours':2909,'balance_is_user_supplied_snapshot':True,
            'billing_unit':'allocated GPUs x Slurm parent-job elapsed seconds /3600; count failures too',
            'separate_from_CPU_core_hour_pool':True,'no_retries':True,'updated_at_utc':now}
    (G/'GPU_BUDGET.json').write_text(json.dumps(budget,indent=2)+'\n')
    plan='''# A100 L01 小规模对照：用户授权记录与执行方案

来源：2026-10-23用户在当前聊天直接转达“审核04补充 A100 VASP-GPU”，选B，批准最多3卡时。这是Codex整理的授权记录，不是另造的Claude审核原件。[用户授权]

- CPU的05f/06f继续原限额；新增GPU只算L01两次：07g_L01_1g=1GPU/1MPI/6CPU/90GiB/1h；08g_L01_2g=2GPU/2MPI/12CPU/180GiB/1h。后者afterok前者，便于失败停止和避免同节点竞争影响测速。[用户授权/inference]
- 目录hzw-a100:~/Jony/runs/01_calib_gpu/。PBE/POSCAR/KPOINTS/ENCUT520/NBANDS1536/EDIFF1E-5/LREALAuto/AMIX0.2/BMIX0.0001与CPU06f相同；GPU并行NCORE=1、KPAR=1，NSIM默认。vasp_gam，mkl/latest、nvhpc/25.3，每GPU一个MPI进程、每进程6线程。[code]
- 三个POTCAR逐个SHA与CPU完全一致；二进制与输入运行前校验；POTCAR原文不回传。脚本内部59分钟截止、10秒强制退出宽限，防止Slurm KillWait导致超过3卡时；不自动重试。[code]
- 比较每SCF步、SCF步数、总墙钟、实际GPU卡时、GPU显存采样，以及同L01对CPU06f的总能和最大逐原子力差。只有COMPLETED且SCF独立通过的输出可比较。能量一致要求绝对差≤0.01eV/胞；力差单位为eV/Å，用户粘贴的0.01eV/胞不能用于力，先如实报告，阈值待审核。[用户授权/inference]
- 1/2GPU性能同时看墙钟和GPU卡时，不能只选最快。通过精度后再重写316原子大胞GPU预算；94原子后续DFT归CPU预算。GPU余额2909是用户提供的快照；3卡时之外没有获批GPU开销。[inference]
- P1、P2及以后暂停，不提交训练、理想格点弛豫或生产DFT。失败先停止后继、保存实耗，不自动重试。[用户授权]

并行设置依据：[VASP官方GPU说明](https://vasp.at/wiki/OpenACC_GPU_port_of_VASP)、[MPI/OpenMP说明](https://vasp.at/wiki/Combining_MPI_and_OpenMP)；管理员/data/software/app/demo_vasp/run.slurm已读取。[direct_text/code]
'''
    (ROOT/'GPU_PILOT_PLAN.md').write_text(plan)
    print(json.dumps({'prepared':[j['name'] for j in jobs],'hard_card_hours':3,'submitted':False},indent=2))

if __name__=='__main__':main()
