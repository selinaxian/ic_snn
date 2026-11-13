"""Verify and optionally restore archived project files, without overwriting changes.

python archives/restore.py --verify-only
python archives/restore.py
"""
import argparse,hashlib,json,tarfile,tempfile,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    for group in manifest['groups']:
        expected={e['path']:e for e in group['files']};seen=set()
        with tempfile.TemporaryFile() as tmp:
            for part in group['parts']:
                path=HERE/part['path'];assert path.stat().st_size==part['bytes'] and sha(path)==part['sha256'],path
                with path.open('rb') as src:shutil.copyfileobj(src,tmp)
            tmp.seek(0)
            with tarfile.open(fileobj=tmp,mode='r|xz') as tar:
                for member in tar:
                    assert member.isfile() and member.name in expected and member.name not in seen,member.name
                    target=(ROOT/member.name).resolve();assert target.is_relative_to(ROOT.resolve()),member.name
                    e=expected[member.name];assert member.size==e['bytes'];h=hashlib.sha256()
                    write=not args.verify_only and not target.exists()
                    if not args.verify_only and target.exists():assert sha(target)==e['sha256'],f'Refusing to overwrite modified file: {target}'
                    if write:target.parent.mkdir(parents=True,exist_ok=True)
                    dest=target.open('xb') if write else None
                    try:
                        src=tar.extractfile(member)
                        for b in iter(lambda:src.read(1024*1024),b''):
                            h.update(b)
                            if dest:dest.write(b)
                    finally:
                        if dest:dest.close()
                    assert h.hexdigest()==e['sha256'],member.name
                    seen.add(member.name)
        assert seen==set(expected)
        print(group['group'],len(seen),'files verified',flush=True)
    print('All archive checks passed.',flush=True)
if __name__=='__main__':main()
