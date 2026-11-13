"""Label-free whole-window familiarity with chronological calibration."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from walk_forward_spike_readout import FOLDS, load_data
from snn_core.encoding import encode_factor_thresholds
from snn_core.snncube import SNNCube


def nearest_correlations(reference, query):
    """Match complete binary rasters by centered cosine (binary Pearson)."""
    ref = reference.astype(np.float32)
    ref -= ref.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ref, axis=1)
    valid = norms > 0
    ref[valid] /= norms[valid,None]
    similarities, indices = [], []
    for start in range(0,len(query),128):
        block = query[start:start+128].astype(np.float32)
        block -= block.mean(axis=1,keepdims=True)
        scale = np.linalg.norm(block,axis=1)
        mask = scale>0
        block[mask] /= scale[mask,None]
        scores = block @ ref.T
        scores[:,~valid] = -np.inf
        idx = scores.argmax(axis=1)
        best = scores[np.arange(len(block)),idx]
        best[~mask] = 0
        idx[~mask] = -1
        similarities.extend(best)
        indices.extend(idx)
    return np.asarray(similarities),np.asarray(indices)


def percentiles(reference, values):
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side='left')
    right = np.searchsorted(ordered, values, side='right')
    return (left+right)/(2*len(ordered))


def main(output):
    sessions, indices, _ = load_data()
    output.mkdir(parents=True)
    results, profiles = [], []
    for fold,(train_end,cal_end,test_end) in enumerate(FOLDS,1):
        active = [(indices[t[0].date()],t,b,x) for t,b,x in sessions if indices[t[0].date()]<=test_end]
        training = np.concatenate([x for day,_,_,x in active if day<=train_end])
        center, scale = np.median(training,axis=0), np.maximum(training.std(axis=0),1e-10)
        encoded = [(day,t,b,encode_factor_thresholds(x,center,scale)) for day,t,b,x in active]
        fixed = SNNCube(11,43,2.5,side=6,depression=1,tau=2)
        learned = SNNCube(11,43,2.5,side=6,depression=1,tau=2)
        for day,_,_,events in encoded:
            if day<=train_end:
                learned.simulate(events,learn=True)
        day_ids, times, realized, past_range = [],[],[],[]
        for day,t,b,events in encoded:
            ends = np.arange(5,len(b))
            labels = np.full(len(ends),np.nan)
            labels[:-1] = 1e4*np.log(b[ends[:-1]+1,3]/b[ends[:-1],3])
            past = b[ends[:,None]+np.arange(-4,1)]
            past_range.extend(1e4*np.log(past[:,:,1].max(axis=1)/past[:,:,2].min(axis=1)))
            times.extend(t[i].isoformat(sep=' ') for i in ends)
            day_ids.extend([day]*len(ends))
            realized.extend(labels)
        days,times,y,past_range = map(np.asarray,(day_ids,times,realized,past_range))
        train, cal, test = days<=train_end,(days>train_end)&(days<=cal_end),days>cal_end
        future = pd.DataFrame(dict(fold=fold,timestamp=times[test],actual_1m_bp=y[test],
                                   past_5m_range_bp=past_range[test]))
        for name,model in [('random',fixed),('stdp',learned)]:
            parts=[]
            for _,_,_,events in encoded:
                spikes=model.simulate(events)
                windows=np.lib.stride_tricks.sliding_window_view(spikes,5,axis=0).transpose(0,2,1)
                parts.append(windows.reshape(len(windows),-1))
            items=np.vstack(parts)
            assert len(items)==len(times)
            similarity, neighbor=nearest_correlations(items[train],items[~train])
            familiarity=percentiles(similarity[cal[~train]],similarity[test[~train]])
            idx=neighbor[test[~train]]
            future[name+'_familiarity']=familiarity
            future[name+'_similarity']=similarity[test[~train]]
            future[name+'_nearest_training_time']=np.where(idx>=0,times[train][idx],'')
            future[name+'_firing_fraction']=items[test].mean(axis=1)
        future['range_percentile']=percentiles(past_range[cal],past_range[test])
        for signal in ('random_familiarity','stdp_familiarity','range_percentile'):
            for band,mask in [('bottom_20',future[signal]<=.2),('top_20',future[signal]>=.8)]:
                group=future[mask & future.actual_1m_bp.notna()]
                clock=future.timestamp.str[11:16]
                profile=future.assign(abs_bp=future.actual_1m_bp.abs()).groupby(clock).abs_bp.mean()
                baseline=group.timestamp.str[11:16].map(profile)
                profiles.append(dict(fold=fold,signal=signal,band=band,minutes=len(group),
                    days=group.timestamp.str[:10].nunique(),
                    median_recent_range_bp=group.past_5m_range_bp.median(),
                    mean_future_abs_bp=group.actual_1m_bp.abs().mean(),
                    abs_minus_clock_mean_bp=(group.actual_1m_bp.abs()-baseline).mean()))
        results.append(future)
        print('whole-window fold',fold,'minutes',len(future),flush=True)
    pd.concat(results,ignore_index=True).to_csv(output/'whole_window_signals.csv',index=False)
    pd.DataFrame(profiles).to_csv(output/'score_profiles.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
