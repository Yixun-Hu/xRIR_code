import json, numpy as np
root='/home/yixunhu/data_cache/HAA_xrir/hallway'
m=json.load(open(f'{root}/meta.json')); xyz=np.load(f'{root}/xyzs.npy'); spk=np.load(f'{root}/speaker_xyz.npy')
rirs=np.load(f'{root}/rirs.npy', mmap_mode='r'); print('rirs shape', rirs.shape, 'sr', m['sr'])
sr=m['sr']
def c50(ir):
    ir=np.asarray(ir,float); e=ir**2
    onset=np.argmax(np.abs(ir)>0.2*np.abs(ir).max())   # direct-path onset
    n50=int(0.05*sr); early=e[onset:onset+n50].sum(); late=e[onset+n50:].sum()+1e-12
    return 10*np.log10(early/late), onset/sr*343, e.sum()
rel=xyz-spk[None]; y=rel[:,1]; dist=np.linalg.norm(rel[:,:2],axis=1)
side=np.where(y>0,'+y','-y')
idx=np.arange(len(xyz))
vals=np.array([c50(rirs[i]) for i in idx])
print('\nGround-truth C50 (dB), direct-path distance from onset (m), total energy, by side and distance bin:')
for lo,hi in [(0,2),(2,4),(4,6),(6,9)]:
    for s in ['-y','+y']:
        b=(dist>=lo)&(dist<hi)&(side==s)
        if b.sum(): print(f'  dist {lo}-{hi} m {s} (n={b.sum():3d}): C50 {vals[b,0].mean():6.2f} +- {vals[b,0].std():4.2f} | onset dist {vals[b,1].mean():5.2f} (geom {dist[b].mean():5.2f}) | energy {vals[b,2].mean():.4f}')
print('\nTrain mics: idx, side, dist, C50'); 
for i in m['train']: print(f'  {i:3d} {side[i]} d={dist[i]:5.2f} C50={vals[i,0]:6.2f} E={vals[i,2]:.4f}')
# depth panorama both directions
for name in ['depth.npy','depth_local.npy']:
    try: d=np.load(f'{root}/{name}')
    except FileNotFoundError: continue
    H,W=d.shape; th=np.degrees((np.arange(W)+0.5)*2*np.pi/W-np.pi)
    print(f'\n{name}: shape {d.shape}, min {d.min():.2f} max {d.max():.2f}, zeros/nans {(~np.isfinite(d)).sum()+(d<=0).sum()}')
    for r in [H//2-32,H//2,H//2+32]:
        row=d[r]; print(f'  row {r}: per-22.5deg-sector median:', ' '.join(f'{np.median(row[(th>=a)&(th<a+22.5)]):5.2f}' for a in np.arange(-180,180,22.5)))
    print('  sector labels:                ', ' '.join(f'{int(a):5d}' for a in np.arange(-180,180,22.5)))
    # 180-degree roll symmetry: compare d with roll by W/2
    dr=np.roll(d,W//2,axis=1); print(f'  180deg-roll relative abs diff (median over pixels): {np.median(np.abs(d-dr)/(np.abs(d)+1e-6)):.3f}; 90deg-roll: {np.median(np.abs(d-np.roll(d,W//4,axis=1))/(np.abs(d)+1e-6)):.3f}')
