"""Expanding walk-forward motif discovery, calibration, and frozen scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from expand_spike_patterns import MAX_EXPANDED, NEURONS, discover
from mine_local_spike_patterns import MIN_SUPPORT
from mine_decision_aligned_patterns import describe, key_at
from snn_core.encoding import encode_factor_thresholds
from snn_core.market_data import load_sessions, ohlcv_features
from snn_core.metrics import evaluate, paired_interval
from snn_core.snncube import SNNCube

FOLDS = ((81,101,121),(101,121,141),(121,141,161))
TOP = 4
MIN_CALIBRATION_DAYS = 5


def load_data():
    history = Path('data/ic_holdout_input.csv')
    fresh = Path('experiments/frozen_patterns_sep_holdout/new_minutes.csv')
    sessions = []
    for path in (history, fresh):
        for stamps,bars in load_sessions(path):
            features = ohlcv_features(bars,True)[:,:6]
            sessions.append((stamps,bars,features))
    dates = sorted({stamps[0].date() for stamps,_,_ in sessions})
    assert len(dates) == 166 and all(left[0][-1] < right[0][0] for left, right in zip(sessions, sessions[1:]))
    return sessions, {day:i+1 for i,day in enumerate(dates)}, dates


def catalog(train):
    half = MAX_EXPANDED//2
    long,_,_ = discover(train,max_expanded=half)
    short,_,_ = discover(train[:,:3*NEURONS],max_lag=2,max_expanded=half)
    return sorted({key_at(child,0) for _,child,_ in long+short})


def main(output):
    sessions, day_index, dates = load_data()
    output.mkdir(parents=True)
    pattern_rows, prediction_rows, metrics = [], [], []
    for fold,(train_end,cal_end,test_end) in enumerate(FOLDS,1):
        active = [(day_index[t[0].date()],t,b,x) for t,b,x in sessions if day_index[t[0].date()]<=test_end]
        training = np.concatenate([x for day,_,_,x in active if day<=train_end])
        center,scale = np.median(training,axis=0),np.maximum(training.std(axis=0),1e-10)
        active = [(day,t,b,encode_factor_thresholds(x,center,scale)) for day,t,b,x in active]
        fixed = SNNCube(11,43,2.5,side=6,depression=1,tau=2)
        learned = SNNCube(11,43,2.5,side=6,depression=1,tau=2)
        for day,_,_,encoded in active:
            if day<=train_end:
                learned.simulate(encoded,learn=True)
        day_id, stamps, labels = [], [], []
        for day,t,b,encoded in active:
            end=np.arange(4,len(encoded))
            next_return=np.full(len(end),np.nan)
            next_return[:-1]=1e4*np.log(b[end[:-1]+2,3]/b[end[:-1]+1,3])
            day_id.extend([day]*len(end))
            stamps.extend(t[i+1].isoformat(sep=' ') for i in end)
            labels.extend(next_return)
        day_id,stamps,y=map(np.asarray,(day_id,stamps,labels))
        train=day_id<=train_end
        cal=(day_id>train_end)&(day_id<=cal_end)&np.isfinite(y)
        test=(day_id>cal_end)&np.isfinite(y)
        test_scores={}
        for pool,model in [('random',fixed),('stdp',learned)]:
            blocks=[]
            for _,_,_,encoded in active:
                spikes=model.simulate(encoded)
                windows=np.lib.stride_tricks.sliding_window_view(spikes,5,axis=0).transpose(0,2,1)
                blocks.append(windows.reshape(len(windows),-1))
            items=np.vstack(blocks)
            patterns=catalog(items[train])
            qualified=[]
            for pattern in patterns:
                hit=items[:,pattern].all(axis=1)
                assert int(hit[train].sum())>=MIN_SUPPORT
                selected=hit&cal
                if selected.sum()>=MIN_SUPPORT and len(set(day_id[selected]))>=MIN_CALIBRATION_DAYS:
                    qualified.append((float(y[selected].mean()),pattern,hit,selected))
            qualified.sort(key=lambda x:(-abs(x[0]),x[1]))
            sums=np.zeros(test.sum())
            counts=np.zeros(test.sum(),dtype=int)
            for rank,(score,pattern,hit,selected) in enumerate(qualified,1):
                triggered=hit[test]
                if rank <= TOP:
                    sums[triggered]+=score
                    counts[triggered]+=1
                pattern_rows.append(dict(fold=fold,pool=pool,pattern=describe(pattern),selection_rank=rank,
                    train_hits=int(hit[train].sum()),cal_hits=int(selected.sum()),
                    cal_days=len(set(day_id[selected])),readout_bp=score,
                    test_hits=int(triggered.sum()),test_days=len(set(day_id[test][triggered])),
                    test_mean_bp=float(y[test][triggered].mean()) if triggered.any() else None))
            scores=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
            test_scores[pool]=(scores,counts)
            traded=counts>0
            direction=np.sign(scores[traded])
            returns=y[test][traded]
            result=evaluate(y[test],scores)
            result.update(fold=fold,pool=pool,train_days=train_end,
                cal_start=str(dates[train_end]),cal_end=str(dates[cal_end-1]),
                test_start=str(dates[cal_end]),test_end=str(dates[test_end-1]),
                candidates=len(patterns),qualified=len(qualified),selected=min(TOP,len(qualified)),
                test_minutes=int(test.sum()),coverage=float(traded.mean()),
                triggered_directional_mean_bp=float((direction*returns).mean()) if traded.any() else None,
                zero_rmse_bp=float(np.sqrt(np.mean(y[test]**2))))
            metrics.append(result)
        random_score,random_count=test_scores['random']
        stdp_score,stdp_count=test_scores['stdp']
        if np.std(random_score)>0 and np.std(stdp_score)>0:
            metrics[-1]['stdp_minus_random_rank_ic_ci']=paired_interval(
                y[test],stdp_score,random_score,day_id[test]).tolist()
        prediction_rows.extend(zip([fold]*int(test.sum()),stamps[test],y[test],
                                   random_score,random_count,stdp_score,stdp_count))
        print('fold',fold,metrics[-2:],flush=True)
    library = pd.DataFrame(pattern_rows)
    library.to_csv(output/'pattern_library.csv',index=False)
    library[library.selection_rank <= TOP].to_csv(output/'selected_patterns.csv',index=False)
    pd.DataFrame(prediction_rows,columns=['fold','timestamp','actual_bp','random_bp','random_matches',
                                         'stdp_bp','stdp_matches']).to_csv(output/'predictions.csv',index=False)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)


