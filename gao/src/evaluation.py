import numpy as np
import pandas as pd

def event_labels(b,window=3):
    rows=[]
    for date,day in b.groupby('trading_date',sort=False):
        changes=[]
        for _,g in day.groupby('segment_id',sort=False):changes.extend(abs(np.diff(g.close.to_numpy())/g.close.to_numpy()[:-1]))
        pivot=float(np.median(changes))
        for _,g in day.groupby('segment_id',sort=False):
            ids=g.index.to_numpy();p=g.close.to_numpy();r=abs(p[1:]/p[:-1]-1)
            for i,bar in enumerate(ids):
                # Literal paper r_t=|P[t+1]/P[t]-1|, strength r[t+1:t+w+1].
                strength=float(r[i+1:i+window+1].mean()) if i+window+1<len(p) else np.nan
                flag=(p[i-window:i].mean()-p[i])*(p[i+1:i+window+1].mean()-p[i]) if i>=window and i+window<len(p) else np.nan
                rows.append(dict(bar_id=int(bar),date=date,pivot_return=pivot,strength=strength,
                    real_spike=float(strength>pivot) if np.isfinite(strength) else np.nan,
                    momentum=float(flag<0) if np.isfinite(flag) and flag!=0 else np.nan))
    return pd.DataFrame(rows)

def summarize_events(signals,labels):
    d=signals.merge(labels.drop(columns='date'),on='bar_id',validate='many_to_one');rows=[]
    for (seed,pool),g in d.groupby(['seed','pool']):
        hit=g[g.event];rows.append(dict(seed=seed,pool=pool,minutes=len(g),events=len(hit),event_fraction=g.event.mean(),
            evaluable_events=int(hit.real_spike.notna().sum()),spike_accuracy=hit.real_spike.mean(),momentum_fraction=hit.momentum.mean(),
            unconditional_real_fraction=g.real_spike.mean(),total_output_spikes=int(g.output_count.sum())))
    return d,pd.DataFrame(rows)
