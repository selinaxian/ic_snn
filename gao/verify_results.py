"""Independent result integrity checks and selected model reload/replay."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import ROOT,load
from src.network import GaoNetwork
from src.encoding import encode

def main(name):
    out=ROOT/name;cfg=json.loads((out/'config.json').read_text());b,x,dates,groups=load()
    splits=pd.read_csv(out/'date_splits.csv');s=pd.read_csv(out/'minute_signals.csv');t=pd.read_csv(out/'cost_scenario_trades.csv')
    assert (splits.train_date<splits.test_date).all()
    assert not s.duplicated(['seed','pool','bar_id']).any()
    assert s.groupby(['seed','pool','date']).size().eq(198).all()
    assert np.array_equal(s.event.to_numpy(),s.output_count.to_numpy()>0)
    assert len(list((out/'models').glob('*.npz')))==len(splits)*len(cfg['seeds'])*2
    errors=[]
    for j in sorted({0,len(splits)//2,len(splits)-1}):
        date=splits.test_date.iloc[j]
        for pool in ['random','stdp']:
            z=np.load(out/f'models/{date}_104_{pool}.npz');n=GaoNetwork(cfg['network'],104)
            n.w_in[0]=z['w_in'];n.w_out[0]=z['w_out'];sp=[]
            for g in groups[date]:
                ev=encode(x[g],g,z['mean'],z['scale'],104,cfg['network']['steps_per_minute'],cfg['encoding'])
                _,o=n.simulate(ev[None]);sp.append(o[0])
            saved=np.load(out/f'traces/{date}_104_{pool}.npz');assert np.array_equal(np.concatenate(sp),saved['output_spikes'])
            ids=saved['bar_ids'];expected=s[(s.seed==104)&(s.pool==pool)&(s.date==date)].set_index('bar_id').loc[ids]
            assert np.array_equal(np.concatenate(sp).sum(1),expected.output_count)
    # Trades use only known signal and same continuous session; no cross-day/contract exit.
    for _,g in t.groupby(['seed','pool','strategy']):
        selected=s[(s.seed==g.seed.iloc[0])&(s.pool==g.pool.iloc[0])&s.event]
        assert set(g.signal_bar)<=set(selected.bar_id)
        assert (g.entry_bar>=g.signal_bar+2).all() and (g.exit_bar>=g.entry_bar).all()
        assert np.array_equal(b.segment_id.iloc[g.signal_bar].to_numpy(),b.segment_id.iloc[g.exit_bar].to_numpy())
        assert np.allclose(g.net_yuan,g.gross_yuan-g.slippage_yuan-g.fee_yuan,atol=1e-7)
        for _,part in g.groupby('date'):
            a=part.sort_values('entry_bar');assert (a.entry_bar.to_numpy()[1:]>=a.exit_bar.to_numpy()[:-1]).all()
    day=pd.read_csv(out/'cost_scenario_daily.csv')
    for key,g in day.groupby(['seed','pool','strategy']):
        a=t[(t.seed==key[0])&(t.pool==key[1])&(t.strategy==key[2])]
        assert abs(a.net_yuan.sum()-g.daily_net.sum())<1e-5
        assert abs(g.capital.iloc[-1]-(cfg['cost_scenario']['capital_yuan']+a.net_yuan.sum()))<1e-5
    paper=pd.read_csv(out/'paper_style_trades.csv');naive=pd.read_csv(out/'random_timing_reference.csv')
    count=paper.groupby(['seed','pool','date','strategy']).size()
    for row in naive.itertuples():assert count.get((row.seed,row.pool,row.date,row.strategy),0)==row.mean_executed_trades
    hashes=json.loads((out/'source_hashes.json').read_text())
    for f,h in hashes.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    result=dict(complete=True,signals=len(s),trades=len(t),sampled_model_replays=6,signals_and_trades_aligned=True,
        no_overlap_or_cross_session_trade=True,cost_arithmetic_and_capital_verified=True,random_timing_counts_equal=True,source_hashes_unchanged=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/baseline');main(p.parse_args().output)
