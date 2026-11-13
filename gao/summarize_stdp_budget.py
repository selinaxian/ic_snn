"""Summaries and a standalone training-trajectory plot for the fixed ablation."""
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from src.data import ROOT


def main():
    p=ROOT/'results/stdp_budget_audit'
    a=pd.read_csv(p/'learning_accounts.csv')
    agg=a.groupby(['variant','layer']).agg(ltp=('ltp','sum'),ltd=('ltd','sum'),zero_ltp=('zero_ltp','sum'),
        clip=('clip','sum'),normalization=('normalization','sum'),initial_sum=('initial_sum','mean'),
        final_sum=('final_sum','mean'),relative_weight_change=('relative_weight_change','mean'))
    agg['ltp_ltd_ratio']=agg.ltp/agg.ltd.replace(0,np.nan)
    agg['zero_fraction_of_ltp']=agg.zero_ltp/agg.ltp.replace(0,np.nan)
    agg.to_csv(p/'pooled_learning_summary.csv')
    s=pd.read_csv(p/'minute_signals.csv')
    assert len(s)==1283040 and not s.duplicated(['seed','variant','bar_id','control','repeat']).any()
    actual=s[s.control.eq('actual')&s['repeat'].eq(0)].copy()
    actual['all_active']=actual.hidden_active.eq(128)
    activity=actual.groupby('variant').agg(hidden_count=('hidden_count','mean'),all_hidden_active=('all_active','mean'),
        output_count=('output_count','mean'),event_rate=('event','mean'))
    activity.to_csv(p/'pooled_activity.csv')
    perday=actual.groupby(['variant','seed','date']).agg(event_unique=('event','nunique'),count_unique=('output_count','nunique'))
    perday['constant_event']=perday.event_unique.eq(1)
    perday.groupby('variant').agg(constant_event_days=('constant_event','sum'),models=('constant_event','size')).to_csv(p/'day_variability.csv')
    c=pd.read_csv(p/'response_comparisons.csv');rows=[]
    for (variant,control),g in c.groupby(['variant','control']):
        den=np.sqrt(g.var_a.sum()*g.var_b.sum())
        rows.append(dict(variant=variant,control=control,hidden_cell_mae=g.hidden_cell_mae.mean(),
            output_count_mae=g.output_count_mae.mean(),output_count_changed=g.output_count_changed.mean(),
            event_changed=g.event_changed.mean(),within_half_output_corr=g.cross.sum()/den if den else np.nan))
    pd.DataFrame(rows).to_csv(p/'pooled_response.csv',index=False)
    t=pd.read_csv(p/'training_trajectory.csv');t['position']=t.half*99+t.minute
    curves=t.groupby(['variant','position']).mean(numeric_only=True)
    curves.to_csv(p/'mean_training_trajectory.csv')
    im=Image.new('RGB',(1400,960),'white');d=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    d.text((40,15),'Fixed STDP weight-budget ablation: mean training trajectories',fill='black',font=font)
    panels=[('hidden_mean_input_sum','Incoming weight sum per hidden neuron'),('output_sum','Total incoming output weight'),
            ('hidden_count','Hidden spikes per minute (128 neurons)'),('output_count','Output spikes per minute')]
    for idx,(col,title) in enumerate(panels):
        x0=85+(idx%2)*680;y0=105+(idx//2)*420;w=560;h=280
        d.text((x0,y0-35),title,fill='black',font=font)
        ymax=float(curves[col].max())*1.1
        for tick in range(5):
            y=y0+h-h*tick/4;val=ymax*tick/4
            d.line((x0,y,x0+w,y),fill='#dddddd');d.text((x0-70,y-9),f'{val:.1f}',fill='black',font=small)
        for variant,color in [('baseline','#c0392b'),('budget','#2471a3')]:
            v=curves.loc[variant,col].to_numpy()
            pts=[(x0+w*i/197,y0+h-h*float(z)/ymax) for i,z in enumerate(v)]
            d.line(pts,fill=color,width=3)
        xx=x0+w*98.5/197;d.line((xx,y0,xx,y0+h),fill='#888888',width=1)
        for tick in [0,49,98,148,197]:
            d.text((x0+w*tick/197-12,y0+h+8),str(tick+1),fill='black',font=small)
        d.text((x0+80,y0+h+35),'Training minute; middle line = lunch reset',fill='black',font=small)
    d.text((100,920),'Red: original STDP     Blue: fixed incoming budgets     Same input, initialization and STDP rule',fill='black',font=font)
    im.save(p/'training_trajectories.png')
    print(agg.to_string());print(activity.to_string());print(pd.DataFrame(rows).to_string(index=False))
    z=json.loads((p/'complete.json').read_text());z['signal_unique_keys_verified']=True
    (p/'complete.json').write_text(json.dumps(z,indent=2),encoding='utf-8')


if __name__=='__main__':main()
