import json
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1]).resolve()
spec = json.loads(manifest.read_text())
job = spec['jobs'][0]
code = Path(job['cwd'])
source = job['command'][job['command'].index('--source') + 1]
subprocess.run([sys.executable, str(code / 'scripts/verify_code_release.py'), str(code), source,
                spec['release_manifest']], check=True)
sys.path.insert(0, str(code))
from unibrain_agent.paired_training import run

run(spec, manifest.parent)
