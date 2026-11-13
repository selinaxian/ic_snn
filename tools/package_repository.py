"""Lossless repository snapshot of large research files; originals are never deleted."""
from pathlib import Path
import hashlib,json,tarfile,time,subprocess,lzma

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'archives'

class Parts:
    def __init__(self,stem):self.stem=stem;self.n=0;self.handle=None;self.size=0;self.paths=[]
    def write(self,b):
        full=len(b)
        while b:
            if self.handle is None or self.size==90*1024*1024:
                if self.handle:self.handle.close()
                self.n+=1;p=OUT/f'{self.stem}.tar.xz.part{self.n:03d}'
                self.handle=p.open('xb');self.paths.append(p);self.size=0
            take=min(len(b),90*1024*1024-self.size);self.handle.write(b[:take]);b=b[take:];self.size+=take
        return full
    def close(self):
        if self.handle:self.handle.close()

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True);start=time.time()
    names=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).split(b'\0')
    selected=[]
    for raw in names:
        if not raw:continue
        name=raw.decode('utf-8');p=ROOT/name
        if name.startswith(('archives/','tools/')):continue
        if p.suffix.lower() in ['.npz','.npy'] or (p.suffix.lower()=='.csv' and p.stat().st_size>=256*1024):selected.append(name)
    groups={}
    for name in sorted(selected):groups.setdefault(name.split('/')[0],[]).append(name)
    manifest={'format':'concatenated xz-compressed tar split at 90 MiB','created':'2026-11-13','groups':[]}
    for group,names in groups.items():
        entries=[{'path':name,'bytes':(ROOT/name).stat().st_size,'sha256':sha(ROOT/name)} for name in names]
        sink=Parts(group)
        with lzma.LZMAFile(sink,mode='w',preset=3) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as tar:
                for name in names:tar.add(ROOT/name,arcname=name,recursive=False)
        sink.close()
        item={'group':group,'files':entries,'parts':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sink.paths]}
        manifest['groups'].append(item)
        (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(group,len(names),'files',sum(e['bytes'] for e in entries),'->',sum(e['bytes'] for e in item['parts']),'bytes',round(time.time()-start),'seconds',flush=True)
    ignore=ROOT/'.gitignore';text=ignore.read_text(encoding='utf-8')
    text+='\n# Large files stored losslessly in archives; see archives/README.md\n'
    text+='\n'.join('/'+name for name in selected)+'\n'
    ignore.write_text(text,encoding='utf-8')
    print('COMPLETE',len(selected),'files',flush=True)

if __name__=='__main__':main()
