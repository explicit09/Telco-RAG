"""Enumerate a pinned HF standards mirror with pagination (metadata only)."""
import argparse
import json
from pathlib import Path
import re
import urllib.request

REVISION = 'a056f6018a7e8e67052aa68a702e272d0ae95d75'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--releases',nargs='+',default=['14','16','17','19'])
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    for release in a.releases:
        if not release.isdigit():
            raise ValueError('numeric release required')
        url=f'https://huggingface.co/api/datasets/GSMA/3GPP/tree/{REVISION}/marked/Rel-{release}?recursive=true&limit=1000'
        files=[]
        seen=set()
        while url:
            if url in seen:
                raise ValueError('pagination loop')
            seen.add(url)
            with urllib.request.urlopen(url,timeout=90) as response:
                rows=json.load(response)
                link=response.headers.get('Link','')
            files.extend(r for r in rows if r['type']=='file' and r['path'].endswith('/raw.md'))
            next_links=re.findall(r'<([^>]+)>;\s*rel="next"',link)
            url=next_links[0] if next_links else None
            if url and not url.startswith('https://huggingface.co/api/datasets/GSMA/3GPP/tree/'):
                raise ValueError('unexpected pagination origin')
        result={'repository':'GSMA/3GPP','revision':REVISION,'release':release,'files':files}
        (a.output/f'release-{release}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'release':release,'documents':len(files),'bytes':sum(r['size'] for r in files),'pages':len(seen)}),flush=True)


if __name__=='__main__':
    main()
