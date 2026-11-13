"""Encoding stability with ONLY STDP spikes entering the readout."""
import json,hashlib
import numpy as np
import pandas as pd
from src.market_data import ROOT,load
from src.network import ContinuousSNN
from run import stream
from audit_utils import fit_predict
from encoding_stability_audit import corr,apply_saved

def main():
    source=ROOT/'results/encoding_stability_audit';out=ROOT/'results/stdp_only_encoding_audit';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'config.json').read_text());bars,x,_,_,folds=load();fold=folds[0]
    (out/'protocol.json').write_text(json.dumps(dict(
        input_to_readout='ONLY full12x8x64 STDP spikes, no raw factors or voltage',
        target='next-minute absolute log close return bp',readout='fixed ridge MSE penalty .01, calibration only; nonnegative',
        reuse='five already trained STDP networks; no additional STDP training',
        common='different trained networks, common baseline encoding, each readout calibrated separately',
        native='different trained networks, corresponding encodings, each readout calibrated separately',
        inference_only='fixed baseline network AND baseline readout, change test encoding only',
        scope='one initialization, first historical fold, five test days'),indent=2),encoding='utf-8')
    groups=[]
    for _,g in bars.groupby('segment_id',sort=False):
        ids=g.index.to_numpy();ids=ids[np.isfinite(x[ids]).all(1)]
        if len(ids) and bars.trading_date.iloc[ids[0]] in fold['validation_dates']+fold['forecast_dates']:groups.append(ids)
    ends=np.concatenate([g[11:] for g in groups]);ix=ends[:,None]+np.arange(-11,1);meta=bars.iloc[ends]
    test=meta.trading_date.isin(fold['forecast_dates']).to_numpy();testgroups=[g for g in groups if bars.trading_date.iloc[g[0]] in fold['forecast_dates']]
    testids=np.concatenate(testgroups);yall=np.full(len(bars),np.nan)
    for g in groups:yall[g[:-1]]=abs(np.log(bars.close.to_numpy()[g[1:]]/bars.close.to_numpy()[g[:-1]]))*1e4
    y=yall[ends];cal=meta.trading_date.isin(fold['validation_dates']).to_numpy()&np.isfinite(y)
    variants=['base','rate_211','rate_307','timing_211','timing_307'];inputs={}
    for v in variants:
        z=np.load(source/f'{v}_input.npz');e=np.zeros((len(bars),8,24),bool);e[z['bar_ids']]=np.unpackbits(z['packed'],axis=2)[:,:,:24];inputs[v]=e
    rows=[];summary=[];reference=None;maxerr=0
    def record(v,probe,p):
        good=np.isfinite(y[test]);r=reference
        summary.append(dict(variant=v,probe=probe,correlation=corr(p,r),rank_correlation=corr(pd.Series(p).rank(),pd.Series(r).rank()),
            mean_prediction=float(p.mean()),prediction_std=float(p.std()),difference_mae_bp=float(abs(p-r).mean()),
            difference_rms_bp=float(np.sqrt(np.mean((p-r)**2))),rmse_bp=float(np.sqrt(np.mean((p[good]-y[test][good])**2)))))
        rows.append(pd.DataFrame(dict(timestamp=meta.datetime[test].astype(str).to_numpy(),variant=v,probe=probe,prediction=p,actual_abs_1m_bp=y[test])))
    for v in variants:
        for probe in ['common','native']:
            m=ContinuousSNN(cfg,104).load(source/f'{v}_stdp_network.npz');w=m.weight.copy();theta=m.theta.copy()
            _,_,s=stream(m,inputs['base' if probe=='common' else v],groups,capture=True,total_rows=len(bars))
            old=np.load(source/f'{v}_stdp_{probe}_test_trace.npz');assert np.array_equal(testids,old['bar_ids']) and np.array_equal(s[testids],old['spikes'])
            assert np.array_equal(w,m.weight) and np.array_equal(theta,m.theta)
            f=s[ix].reshape(len(ends),-1);assert f.shape[1]==6144
            path=out/f'{v}_{probe}_readout.npz';p=fit_predict([f],cal,test,y,path)
            rebuilt=apply_saved(path,[f[test]]);err=float(abs(p-rebuilt).max());maxerr=max(maxerr,err);assert err<1e-10
            if reference is None:reference=p.copy()
            if v=='base':assert np.array_equal(p,reference)
            record(v,probe,p)
        if v!='base':
            m=ContinuousSNN(cfg,104).load(source/'base_stdp_network.npz')
            _,_,s=stream(m,inputs[v],testgroups,capture=True,total_rows=len(bars))
            p=apply_saved(out/'base_common_readout.npz',[s[ix[test]].reshape(test.sum(),-1)])
            record(v,'inference_only',p)
        print(v,'complete',flush=True)
    d=pd.concat(rows,ignore_index=True);d.to_csv(out/'minute_predictions.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    assert len(d)==12320 and not d.duplicated(['variant','probe','timestamp']).any()
    (out/'verification.json').write_text(json.dumps(dict(complete=True,readout_features=6144,raw_features_in_readout=0,
        all_saved_test_traces_exact=True,baseline_common_native_exact=True,prediction_reconstruction_max_error=maxerr,
        test_minutes=int(test.sum()),test_labeled=int(np.isfinite(y[test]).sum()),
        source_hash=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')

if __name__=='__main__':main()
