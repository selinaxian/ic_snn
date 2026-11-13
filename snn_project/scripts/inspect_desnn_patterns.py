from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run_diagnostics import load_features
from snn_core.desnn import desnn_signature

base=Path('experiments')
out=base/'desnn_pattern_inspection'
out.mkdir(exist_ok=True)
_,_,_,spikes,dates,groups,y=load_features(Path('data/ic_holdout_input.csv'),base/'segmented_v1')
train=spikes[groups==0]
query=spikes[groups!=0]
with np.load(base/'desnn_neighbor_variance/fixed_neighbors.npz') as f:
    np.testing.assert_array_equal(f['train_timestamps'],dates[groups==0])
    np.testing.assert_array_equal(f['query_timestamps'],dates[groups!=0])
    indices=f['stdp_pool_indices']
    weights=f['stdp_pool_weights']
signature=np.stack([desnn_signature(a,.5) for a in spikes])
ref=signature[groups==0]
q=signature[groups!=0]
distance=np.linalg.norm(ref[indices]-q[:,None,:],axis=2)
jaccard=np.empty_like(distance)
active=np.empty_like(distance)
for i in range(0,len(query),128):
    span=slice(i,i+128)
    a=query[span,None,:,:]
    b=train[indices[span]]
    jaccard[span]=(a&b).sum(axis=(2,3))/np.maximum((a|b).sum(axis=(2,3)),1)
    aa=a.any(axis=2)
    bb=b.any(axis=2)
    active[span]=(aa&bb).sum(axis=2)/np.maximum((aa|bb).sum(axis=2),1)
test=np.flatnonzero(groups[groups!=0]==2)
near=distance.argmin(axis=1)
test=test[np.argsort(distance[test,near[test]])]
rng=np.random.default_rng(43)
random_idx=rng.integers(len(train),size=len(test))
random_d=np.linalg.norm(q[test]-ref[random_idx],axis=1)
random_j=[]
for i,j in zip(test,random_idx):
    a,b=query[i],train[j]
    random_j.append((a&b).sum()/max((a|b).sum(),1))
with (out/'pair_similarity.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.writer(f)
    w.writerow(['query','neighbor','weight','signature_euclidean','spike_jaccard','active_neuron_jaccard','query_return_bp','neighbor_return_bp'])
    for i in test:
        for n in np.argsort(distance[i]):
            idx=indices[i,n]
            w.writerow([dates[groups!=0][i],dates[groups==0][idx],weights[i,n],distance[i,n],jaccard[i,n],active[i,n],y[groups!=0][i],y[groups==0][idx]])
def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
def draw_case(i,name):
    img=Image.new('RGB',(1530,650),'#f7f9fc')
    d=ImageDraw.Draw(img)
    d.text((35,23),'IC | deSNN spike raster: '+name,fill='#14233a',font=font(30))
    d.text((35,64),'Same neuron order and 30-minute grid across panels. Blue: shared with query. Orange: neighbor only.',fill='#34455e',font=font(19))
    seq=[(query[i],dates[groups!=0][i],y[groups!=0][i],None)]
    for n in np.argsort(distance[i]):
        idx=indices[i,n]
        seq.append((train[idx],dates[groups==0][idx],y[groups==0][idx],n))
    for k,(a,t,r,n) in enumerate(seq):
        left=52+378*k
        top=165
        d.text((left,106),('Query' if n is None else f'Neighbor {k}')+'  |  '+t[5:16],fill='#14233a',font=font(18))
        d.text((left,133),f'Next 5 min return: {r:+.1f} bp',fill='#34455e',font=font(16))
        d.rectangle((left,top,left+300,top+432),fill='white',outline='#bcc9db')
        for t0 in (5,10,15,20,25):
            d.line((left+10*t0,top,left+10*t0,top+432),fill='#e1e7ef')
        for t0,j in np.argwhere(a):
            x=left+10*t0+3
            y0=top+2*j
            color='#2872bf' if n is None or query[i,t0,j] else '#dd6b27'
            d.rectangle((x,y0,x+4,y0+1),fill=color)
        d.text((left,604),'-29 min',fill='#34455e',font=font(14))
        d.text((left+265,604),'0 min',fill='#34455e',font=font(14))
        if n is not None:
            d.text((left,625),f'd={distance[i,n]:.3f} | overlap={jaccard[i,n]:.1%} | w={weights[i,n]:.2f}',fill='#14233a',font=font(15))
    img.save(out/(name+'.png'))
for p,name in [(0.05,'close_match'),(.5,'median_match'),(.95,'distant_match')]:
    i=test[round(p*(len(test)-1))]
    draw_case(i,name)
    print(name,dates[groups!=0][i],distance[i],jaccard[i],active[i])
for name,v in [('nearest distance',distance[test,near[test]]),('nearest spike overlap',jaccard[test,near[test]]),('nearest active overlap',active[test,near[test]]),('uniform distance',random_d),('uniform spike overlap',random_j)]:
    print(name,np.quantile(v,[.05,.5,.95]))
