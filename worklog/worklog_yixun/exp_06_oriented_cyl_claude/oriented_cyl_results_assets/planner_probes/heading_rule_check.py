import json, math, numpy as np
root='/home/yixunhu/data_cache/HAA_xrir'
CANDS={'+x':0.0,'+y':90.0,'-x':180.0,'-y':-90.0}
def level(h, sr, w):
    h=np.asarray(h,float); n0=int(np.argmax(np.abs(h)>0.2*np.abs(h).max())); n=int(round(w*sr)); return float((h[n0:n0+n]**2).sum())
def wrap(d): return (d+180)%360-180
def decide(th_deg, L):  # L: dict window->array
    out={}
    for name,c in CANDS.items():
        inside=np.abs(wrap(th_deg-c))<=45
        if inside.sum()<3 or (~inside).sum()<3: out[name]=None; continue
        out[name]={w: float(L[w][inside].mean()-L[w][~inside].mean()) for w in L}
    ev={k:v for k,v in out.items() if v is not None}
    if not ev: return None, out, 'no evaluable candidate'
    winners={w: max(ev, key=lambda k: ev[k][w]) for w in L}
    if len(set(winners.values()))>1: return None, out, f'windows disagree {winners}'
    win=winners['5ms']
    for w in L:
        if ev[win][w] < 3: return None, out, f'contrast {ev[win][w]:.1f} dB < 3 dB ({w})'
        for k,v in ev.items():
            if k!=win and ev[win][w]-v[w] < 3: return None, out, f'margin over {k} {ev[win][w]-v[w]:.1f} dB < 3 dB ({w})'
    return win, out, 'ok'
for room in ['class_room','dampened_room','hallway','complex_room']:
    m=json.load(open(f'{root}/{room}/meta.json')); xyz=np.load(f'{root}/{room}/xyzs.npy'); spk=np.load(f'{root}/{room}/speaker_xyz.npy'); rirs=np.load(f'{root}/{room}/rirs.npy',mmap_mode='r')
    tr=np.array(m['train']); rel=xyz[tr]-spk[None]; th=np.degrees(np.arctan2(rel[:,1],rel[:,0])); d=np.linalg.norm(rel[:,:2],axis=1)
    L={w: np.array([10*math.log10(level(rirs[i], m['sr'], s)*dd**2) for i,dd in zip(tr,d)]) for w,s in [('5ms',0.005),('50ms',0.05)]}
    win, table, why = decide(th, L)
    loo=[decide(np.delete(th,j), {w: np.delete(L[w],j) for w in L})[0] for j in range(len(tr))]
    stable = win is not None and all(x==win for x in loo)
    fmt={k:(None if v is None else {w: round(x,1) for w,x in v.items()}) for k,v in table.items()}
    print(f'{room:14s} decision {win} ({why}); leave-one-out winners {sorted(set(map(str,loo)))} stable={stable}; contrasts dB {fmt}')
