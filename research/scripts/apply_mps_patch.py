"""Apply only the recorded MACE 0.3.16 auxiliary-output compatibility change."""
import hashlib, importlib.util, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
record=json.loads((root/'sources/mace_patch_manifest.json').read_text())
spec=importlib.util.find_spec('mace')
p=Path(list(spec.submodule_search_locations)[0])/'modules/models.py'
data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
if digest==record['patched_sha256']:
    print('Recorded MPS compatibility patch is already applied.')
elif digest==record['original_sha256']:
    old=b'node_energy = node_e0.clone().double() + node_inter_es.clone().double()'
    new=b'node_energy = node_e0.clone() + node_inter_es.clone()  # YSZ probe: MPS float32 compatibility'
    assert data.count(old)==1
    modified=data.replace(old,new)
    assert hashlib.sha256(modified).hexdigest()==record['patched_sha256']
    p.write_bytes(modified)
    print('Recorded patch applied; checkpoint weights are unchanged.')
else:
    raise SystemExit('Unrecognized MACE source; no files changed. Use the locked package version.')
