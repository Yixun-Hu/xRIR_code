import json, numpy as np, os
root='/home/yixunhu/data_cache/HAA_xrir'
def load(room, model, run):
    p=f'ckpt/sim2real/{model}/{run}/per_sample_{room}.json' if run=='zeroshot' else f'ckpt/sim2real/{model}/{run}/eval/per_sample_{room}.json'
    return json.load(open(p))
for room in ['hallway','class_room','complex_room','dampened_room']:
    m=json.load(open(f'{root}/{room}/meta.json'))
    xyz=np.load(f'{root}/{room}/xyzs.npy'); spk=np.load(f'{root}/{room}/speaker_xyz.npy')
    test=np.array(m['test']); train=np.array(m['train'])
    rel=xyz[test]-spk[None]; dist=np.linalg.norm(rel[:,:2],axis=1); az=np.degrees(np.arctan2(rel[:,1],rel[:,0]))
    print(f'\n===== {room}: speaker {spk.round(2)}, n_test={len(test)}; mic xyz range min {xyz.min(0).round(2)} max {xyz.max(0).round(2)}')
    print(' train mics rel to speaker (x,y):', np.round((xyz[train]-spk[None])[:,:2],2).tolist())
    d=np.load(f'{root}/{room}/depth.npy'); H,W=d.shape
    row=d[H//2]; cols=np.arange(W); th=np.degrees((cols+0.5)*2*np.pi/W-np.pi)
    # depth at horizon per 45 deg sector
    print(' horizon depth per 45deg sector (deg: median m):', {int(a): float(np.median(row[(th>=a)&(th<a+45)]).round(2)) for a in range(-180,180,45)})
    for run in ['zeroshot','seed0','seed1','seed2']:
        try:
            c=load(room,'cyl',run); s=load(room,'control',run)
        except FileNotFoundError: continue
        assert c['index']==s['index']==test.tolist(), (run)
        cc=np.array(c['c50'],float); sc=np.array(s['c50'],float); ce=np.array(c['edt'],float); se=np.array(s['edt'],float)
        ok=np.isfinite(cc)&np.isfinite(sc)
        dc=cc-sc; de=ce-se
        print(f' [{run}] C50 mean cyl {cc[ok].mean():.3f} ctrl {sc[ok].mean():.3f} diff {dc[ok].mean():+.3f} | EDT diff {de[ok].mean():+.4f}')
        # by distance tercile
        qs=np.quantile(dist,[0,1/3,2/3,1])
        for i in range(3):
            b=(dist>=qs[i])&(dist<=qs[i+1])&ok
            print(f'    dist {qs[i]:5.2f}-{qs[i+1]:5.2f} m (n={b.sum():3d}): C50 cyl {cc[b].mean():.3f} ctrl {sc[b].mean():.3f} diff {dc[b].mean():+.3f} | EDT cyl {ce[b].mean():.4f} ctrl {se[b].mean():.4f} diff {de[b].mean():+.4f}')
        # by azimuth sector (room frame)
        for a in range(-180,180,45):
            b=(az>=a)&(az<a+45)&ok
            if b.sum()>=5: print(f'    az [{a:4d},{a+45:4d}) (n={b.sum():3d}): C50 diff {dc[b].mean():+.3f} EDT diff {de[b].mean():+.4f}')
