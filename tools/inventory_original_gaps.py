"""Compare pinned original standards against the parsed mirror inventory."""
import argparse,json,re,urllib.request
from pathlib import Path
from inventory_mirror import REVISION


def main():
    p=argparse.ArgumentParser()
    p.add_argument('parsed_inventory',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--releases',nargs='+',default=['14','16','17','19'])
    p.add_argument('--docx-manifest',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summary=[]
    for release in a.releases:
        if not release.isdigit():raise ValueError('numeric release required')
        if a.docx_manifest:
            if len(a.releases)!=1:raise ValueError('DOCX comparison requires one explicit release')
            parsed=json.loads(a.docx_manifest.read_text())
            specs={Path(row['path']).stem.rsplit('-',1)[0] for row in parsed['files']}
        else:
            parsed=json.loads((a.parsed_inventory/f'release-{release}.json').read_text())
            if parsed['revision']!=REVISION:raise ValueError('mismatched mirror revision')
            specs={Path(row['path']).parent.name for row in parsed['files']}
        url=f'https://huggingface.co/api/datasets/GSMA/3GPP/tree/{REVISION}/original/Rel-{release}?recursive=true&limit=1000'
        files=[];seen=set()
        while url:
            if url in seen:raise ValueError('pagination loop')
            seen.add(url)
            with urllib.request.urlopen(url,timeout=90) as response:
                rows=json.load(response);link=response.headers.get('Link','')
            files.extend(row for row in rows if row['type']=='file')
            next_links=re.findall(r'<([^>]+)>;\s*rel="next"',link)
            url=next_links[0] if next_links else None
            if url and not url.startswith('https://huggingface.co/api/datasets/GSMA/3GPP/tree/'):
                raise ValueError('unexpected pagination origin')
        gaps=[row for row in files if Path(row['path']).stem.rsplit('-',1)[0] not in specs]
        result={'repository':'GSMA/3GPP','revision':REVISION,'release':release,'files':gaps,
                'comparison_repository':parsed['repository'],'comparison_revision':parsed['revision'],
                'method':'Original filename specification prefix missing from existing DOCX prefixes or parsed raw.md parent directory names.'}
        (a.output/f'release-{release}.json').write_text(json.dumps(result,indent=2)+'\n')
        record={'release':release,'original_files':len(files),'parsed_specs':len(specs),'missing_files':len(gaps),'missing_bytes':sum(row['size'] for row in gaps),'extensions':sorted({Path(row['path']).suffix for row in gaps})}
        summary.append(record);print(json.dumps(record),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
