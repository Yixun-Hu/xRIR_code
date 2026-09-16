import json, numpy as np
root='/home/yixunhu/data_cache/HAA_xrir'
def c50(ir,sr):
    ir=np.asarray(ir,float); e=ir**2; onset=np.argmax(np.abs(ir)>0.2*np.abs(ir).max()); n=int(0.05*sr)
    return 10*np.log10(e[onset:onset+n].sum()/(e[onset+n:].sum()+1e-12)), e.sum()
for room in ['class_room','complex_room','dampened_room']:
    m=json.load(open(f'{root}/{room}/meta.json')); xyz=np.load(f'{root}/{room}/xyzs.npy'); spk=np.load(f'{root}/{room}/speaker_xyz.npy'); rirs=np.load(f'{root}/{room}/rirs.npy',mmap_mode='r')
    rel=xyz-spk[None]; az=np.degrees(np.arctan2(rel[:,1],rel[:,0])); dist=np.linalg.norm(rel[:,:2],axis=1)
    v=np.array([c50(rirs[i],m['sr']) for i in range(len(xyz))])
    print(f'{room}: GT C50 / energy by 90deg azimuth sector (room frame), mics 1.5-4 m:')
    for a in range(-180,180,90):
        b=(az>=a)&(az<a+90)&(dist>1.5)&(dist<4)
        if b.sum()>=5: print(f'   az [{a:4d},{a+90:4d}) n={b.sum():3d}  C50 {v[b,0].mean():6.2f} +- {v[b,0].std():4.2f}  energy {v[b,1].mean():.4f}')
