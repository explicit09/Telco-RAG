"""Select missing numbered standards from a downloaded source inventory."""
import argparse,hashlib,json
from pathlib import Path
from inventory_original_gaps import specification_id


def main():
    p=argparse.ArgumentParser()
    p.add_argument('download_manifest',type=Path)
    p.add_argument('baseline_inventory',type=Path)
    a=p.parse_args()
    downloaded=json.loads(a.download_manifest.read_text())
    baseline=json.loads(a.baseline_inventory.read_text())
    existing={specification_id(Path(row['path']).parent.name if row['path'].endswith('/raw.md') else row['path']) for row in baseline['files']}
    kept=[];excluded=[]
    for row in downloaded['files']:
        spec=specification_id(row['path'])
        if spec is None or spec in existing:
            excluded.append({'path':row['path'],'reason':'non-numbered attachment' if spec is None else 'specification already in baseline'})
        else:
            kept.append(row)
    result={**downloaded,'files':kept,'excluded':excluded,
            'download_manifest_sha256':hashlib.sha256(a.download_manifest.read_bytes()).hexdigest(),
            'baseline_inventory_sha256':hashlib.sha256(a.baseline_inventory.read_bytes()).hexdigest(),
            'selection':'All missing numbered specifications; split sections grouped by canonical specification ID. No benchmark questions or answers used.'}
    path=a.download_manifest.with_name('standards.manifest.json')
    with path.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({'kept_files':len(kept),'distinct_specifications':len({specification_id(r['path']) for r in kept}),'excluded_files':len(excluded)}))


if __name__=='__main__':main()
