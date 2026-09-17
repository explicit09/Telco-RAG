"""Sequential, resumable download/index orchestration for missing releases."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('workspace',type=Path)
    a=p.parse_args()
    work=a.workspace.resolve()
    for release in ('17','14','16','19'):
        if shutil.disk_usage(work).free < 6 * 1024**3:
            raise RuntimeError('Less than 6 GiB free; stop before downloading another release')
        subprocess.run([sys.executable,str(ROOT/'download_corpus.py'),str(work/f'mirror-inventory/release-{release}.json'),str(work/f'corpora/gsma-r{release}')],check=True)
        subprocess.run([sys.executable,str(ROOT/'index_corpus.py'),str(work/f'corpora/gsma-r{release}/manifest.json'),str(work/f'indexes/3gpp-r{release}.sqlite'),'--corpus',f'3gpp-r{release}','--release',release],check=True)
        print(f'RELEASE {release} READY',flush=True)


if __name__=='__main__':
    main()
