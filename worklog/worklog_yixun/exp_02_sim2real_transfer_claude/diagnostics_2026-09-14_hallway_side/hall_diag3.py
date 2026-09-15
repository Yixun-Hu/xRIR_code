import json, numpy as np
root='/home/yixunhu/data_cache/HAA_xrir'
print('180-degree roll symmetry of the depth panorama used (mean relative abs diff; 90-degree roll for contrast):')
for room in ['hallway','class_room','complex_room','dampened_room']:
    d=np.load(f'{root}/{room}/depth.npy'); W=d.shape[1]
    r180=np.mean(np.abs(d-np.roll(d,W//2,axis=1))/(np.abs(d)+1e-6)); r90=np.mean(np.abs(d-np.roll(d,W//4,axis=1))/(np.abs(d)+1e-6))
    print(f'  {room:14s} 180deg: {r180:.4f}   90deg: {r90:.4f}   max|diff| 180deg: {np.abs(d-np.roll(d,W//2,axis=1)).max():.3f} m')
m=json.load(open(f'{root}/hallway/meta.json')); xyz=np.load(f'{root}/hallway/xyzs.npy'); spk=np.load(f'{root}/hallway/speaker_xyz.npy')
test=np.array(m['test']); y=(xyz[test]-spk[None])[:,1]; side=np.where(y>0,'+y','-y')
print('\nHallway per-side absolute errors (mean over test mics on that side):')
for run in ['zeroshot','seed0','seed1','seed2']:
    out=[]
    for model in ['cyl','control','released']:
        p=f'ckpt/sim2real/{model}/{run}/per_sample_hallway.json' if run=='zeroshot' else f'ckpt/sim2real/{model}/{run}/eval/per_sample_hallway.json'
        try: d=json.load(open(p))
        except FileNotFoundError: continue
        assert d['index']==test.tolist()
        c=np.array(d['c50'],float); e=np.array(d['edt'],float); t=np.array(d['t60'],float)
        for s in ['-y','+y']:
            b=side==s
            out.append(f'{model:8s} {s}: C50 {np.nanmean(c[b]):5.2f} dB  EDT {np.nanmean(e[b]):.4f} s  T60 {np.nanmean(t[b]):5.2f} %')
    print(f' [{run}]'); [print('    '+o) for o in out]
