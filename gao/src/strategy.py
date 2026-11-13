"""Paper-style price-point ledger and separate constrained IC cost scenario."""
import numpy as np
import pandas as pd

NAMES=['momentum','alexander','oscillator']

def directions(prices,i):
    if i<3:return [0,0,0]
    past=prices[i-3:i];mom=int(np.sign(prices[i]-past.mean()));alf=int(np.sign(prices[i]-prices[i-1]))
    lo,hi=past.min(),past.max();osc=0 if hi==lo else int(np.sign((prices[i]-lo)/(hi-lo)*100-50))
    return [mom,alf,osc]

def paper_candidates(b,signal_ids,holding=3):
    rows=[]
    for _,g in b.groupby('segment_id',sort=False):
        ids=g.index.to_numpy();price=g.close.to_numpy();active=set(signal_ids)&set(ids)
        for i,bar in enumerate(ids):
            if bar not in active or i+2>=len(ids):continue
            entry=i+1;exit=min(entry+holding,len(ids)-1)
            for strategy,direction in zip(NAMES,directions(price,i)):
                if direction==0:continue
                rows.append(dict(strategy=strategy,date=str(g.trading_date.iloc[0]),contract=str(g.contract_id.iloc[0]),
                    signal_bar=int(bar),entry_bar=int(ids[entry]),exit_bar=int(ids[exit]),direction=direction,
                    entry_price=float(price[entry]),exit_price=float(price[exit]),
                    gross_return=direction*(price[exit]/price[entry]-1),shortened=exit-entry<holding))
    return rows

def executable_trades(b,signal_ids,cfg,holding=3):
    cost=cfg['cost_scenario'];records=[];daily=[]
    for strategy in NAMES:
        capital=cost['capital_yuan'];peak=capital
        for date,day in b.groupby('trading_date',sort=False):
            opening=capital;n=0
            for _,g in day.groupby('segment_id',sort=False):
                ids=g.index.to_numpy();p=g.close.to_numpy();eligible=set(signal_ids)&set(ids);busy=-1
                for i,bar in enumerate(ids):
                    if bar not in eligible or i<busy or i+2>=len(ids):continue
                    direction=directions(p,i)[NAMES.index(strategy)]
                    if not direction:continue
                    entry=i+2;exit=entry+holding
                    # One-minute processing allowance; t+2 open follows t+1 close.
                    entryprice=float(g.open.iloc[entry]);exitprice=float(g.open.iloc[exit]) if exit<len(ids) else float(g.close.iloc[-1])
                    exitidx=min(exit,len(ids)-1)
                    margin=entryprice*cost['multiplier']*cost['margin_fraction']
                    if capital<margin:continue
                    slip=cost['tick_size']*cost['slippage_ticks_per_side']
                    fillentry=entryprice+direction*slip;fillexit=exitprice-direction*slip
                    gross=direction*(exitprice-entryprice)*cost['multiplier']
                    slippage=2*slip*cost['multiplier']
                    fee=(fillentry*(cost['open_fee_rate']+cost['broker_extra_rate'])+fillexit*(cost['close_today_fee_rate']+cost['broker_extra_rate']))*cost['multiplier']+2*cost['order_fee_yuan_per_side']
                    net=gross-slippage-fee;capital+=net;n+=1;busy=exitidx
                    records.append(dict(strategy=strategy,date=date,contract=g.contract_id.iloc[0],signal_bar=int(bar),entry_bar=int(ids[entry]),exit_bar=int(ids[exitidx]),
                        direction=direction,quantity=1,reference_entry=entryprice,reference_exit=exitprice,entry_fill=fillentry,exit_fill=fillexit,
                        gross_yuan=gross,slippage_yuan=slippage,fee_yuan=fee,net_yuan=net,capital_after=capital,shortened=exit>=len(ids)))
            peak=max(peak,capital);daily.append(dict(strategy=strategy,date=date,capital=capital,daily_net=capital-opening,daily_return=(capital-opening)/opening,
                closed_equity_drawdown=capital/peak-1,trades=n))
    return records,daily

def random_timing_reference(b,minute_signals,repeats,seed,holding):
    rng=np.random.default_rng(seed);rows=[]
    for date,g in minute_signals.groupby('date',sort=False):
        ids=g.bar_id.to_numpy(int);n=int(g.event.sum());day=b[b.trading_date==date]
        candidates=paper_candidates(day,ids,holding);hits=set(g.loc[g.event,'bar_id'])
        for k in NAMES:
            options=[r for r in candidates if r['strategy']==k];count=sum(r['signal_bar'] in hits for r in options)
            rr=np.array([r['gross_return'] for r in options]);sums=[float(rr[rng.choice(len(rr),count,replace=False)].sum()) for _ in range(repeats)]
            rows.append(dict(date=date,strategy=k,signal_count=n,repeats=repeats,
                mean_trade_return_sum=float(np.mean(sums)),std_trade_return_sum=float(np.std(sums)),mean_executed_trades=count))
    return rows
