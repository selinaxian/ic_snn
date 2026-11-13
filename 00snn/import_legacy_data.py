"""One-time read-only import; the runnable project then needs no old Python code."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.dependencies'))
import numpy as np
import pandas as pd


def main():
    old=ROOT.parent/'snn_spikedetect/independent_study/results'
    out=ROOT/'data';out.mkdir(exist_ok=False)
    source={'bars':old/'v1/bars_features.parquet','samples':old/'v2/samples.parquet',
            'labels':old/'v2/dataset.npz','folds':old/'v2/folds.json'}
    pd.read_parquet(source['bars']).to_csv(out/'bars.csv',index=False)
    # Future states are not copied into input metadata.
    keep=['sample_id','bar_id','segment_id','contract_id','trading_date','datetime','window_start','entry_bar_id','exit_bar_id']
    pd.read_parquet(source['samples'])[keep].to_csv(out/'samples.csv',index=False)
    with np.load(source['labels']) as a:np.savez_compressed(out/'reference_labels.npz',y=a['y'][:,:3])
    shutil.copyfile(source['folds'],out/'folds.json')
    manifest={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in source.items()}
    (out/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Imported read-only data snapshot',flush=True)


if __name__=='__main__':main()
