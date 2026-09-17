"""Prepare legacy Word/strictly repairable DOCX sources without changing originals."""
import argparse,hashlib,json,re,subprocess,tempfile,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_orphan_revision_closings(xml):
    text=xml.decode('utf-8');stack=[];removed=[]
    pattern=re.compile(r'</?([A-Za-z_][\w.:-]*)(?:\s[^<>]*?)?/?>')
    for match in pattern.finditer(text):
        token,tag=match[0],match[1]
        if token.startswith('</'):
            if stack and stack[-1]==tag:stack.pop()
            elif tag in ('w:ins','w:del') and tag not in stack:removed.append(match.span())
            else:raise ValueError('XML damage is not an orphan revision closing tag')
        elif not token.endswith('/>'):stack.append(tag)
    if not removed:raise ValueError('no supported XML repair found')
    pieces=[];offset=0
    for start,end in removed:
        pieces.append(text[offset:start]);offset=end
    pieces.append(text[offset:])
    result=''.join(pieces).encode('utf-8');ET.fromstring(result)
    return result,len(removed)


def prepare(source,target):
    if source.suffix.lower()=='.doc':
        converter=Path('/usr/bin/textutil')
        if not converter.is_file():raise RuntimeError('legacy DOC preparation requires macOS textutil')
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as folder:
            fresh=Path(folder)/'converted.docx'
            subprocess.run([str(converter),'-convert','docx','-output',str(fresh),str(source)],check=True,capture_output=True,timeout=120)
            with zipfile.ZipFile(fresh) as archive:ET.fromstring(archive.read('word/document.xml'))
            fresh.replace(target)
        return {'method':'textutil DOC to DOCX','converter_sha256':digest(converter)}
    if source.suffix.lower()!='.docx':raise ValueError('unsupported source format')
    with zipfile.ZipFile(source) as archive:
        xml=archive.read('word/document.xml')
        try:ET.fromstring(xml);return None
        except ET.ParseError:repaired,count=repair_orphan_revision_closings(xml)
        target.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as output:
            for entry in archive.infolist():
                output.writestr(entry,repaired if entry.filename=='word/document.xml' else archive.read(entry.filename))
    return {'method':'remove orphan w:ins/w:del closing tags only','removed_tags':count}


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args()
    original=json.loads(a.manifest.read_text());root=a.manifest.parent;files=[];converted=0
    output=a.manifest.with_name('ingest.manifest.json')
    if output.exists():raise FileExistsError('prepared manifest already exists')
    for row in original['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe manifest path')
        source=root/relative
        if digest(source)!=row['sha256']:raise ValueError('source checksum mismatch')
        prepared=Path('prepared')/relative.with_suffix('.docx')
        conversion=prepare(source,root/prepared)
        if conversion:
            converted+=1
            files.append({**row,'path':str(prepared),'sha256':digest(root/prepared),'size':(root/prepared).stat().st_size,
                          'original_path':str(relative),'original_sha256':row['sha256'],'preparation':conversion})
        else:files.append(row)
    result={**original,'files':files,'preparation_source_manifest_sha256':digest(a.manifest),
            'preparation_tool_sha256':digest(Path(__file__)),'prepared_files':converted}
    with output.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({'files':len(files),'prepared_files':converted,'manifest':str(output)}))


if __name__=='__main__':main()
