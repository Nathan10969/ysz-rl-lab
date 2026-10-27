"""Copy review and accounting summaries to project roots, never submit jobs."""
import io,json,tarfile
from pathlib import Path
from submit_claude05 import ssh
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'calibration';G=ROOT/'calibration_gpu'
targets=[('hzw-cpu','/online1/ycsc_hzw/hd100371/Jony/runs/01_calib',C/'BUDGET.json'),
         ('your-cluster','/CLUSTER_HOME/runs/01_calib_gpu',G/'GPU_BUDGET.json')]
for host,remote,budget in targets:
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as tar:
        for file in [C/'REVIEW_STATUS.json',budget,ROOT/'REVIEW_CLAUDE_06_release.md',ROOT/'CLAUDE06_EXECUTION.md']:
            tar.add(file,arcname=file.name,recursive=False)
    ssh(host,'cd '+remote+' && tar -xf -',payload.getvalue())
    print(host+' control summaries synchronized')
