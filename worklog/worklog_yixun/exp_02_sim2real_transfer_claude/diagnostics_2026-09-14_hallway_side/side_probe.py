"""CPU probe: does each model's geometry branch / reference weighting distinguish the two ends of the hallway?"""
import sys, json, numpy as np, torch, torch.nn.functional as F
# keep the probe off the GPUs: the model's shift_and_align calls .cuda() unconditionally
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
_to = torch.Tensor.to
def _to_cpu(self, *a, **k):
    a = tuple(x for x in a if not (isinstance(x, (str, torch.device)) and 'cuda' in str(x)))
    k = {kk: v for kk, v in k.items() if not (kk == 'device' and 'cuda' in str(v))}
    return _to(self, *a, **k)
torch.Tensor.to = _to_cpu
sys.path.insert(0, '.')
torch.set_num_threads(max(4, torch.get_num_threads()))
from model.xRIR_cyl import build_xrir
import sim_to_real.haa_dataset as hd
DS = [c for c in vars(hd).values() if isinstance(c, type) and issubclass(c, torch.utils.data.Dataset) and c is not torch.utils.data.Dataset][0]
ds = DS(['hallway'], 'test', num_shot=8, eval_seed=0)
D = ds.data['hallway']
items = ds.items
meta = json.load(open('/home/yixunhu/data_cache/HAA_xrir/hallway/meta.json'))
train = np.array(meta['train']); test = np.array(meta['test'])
src_all = D['src_local'].numpy()
y_test = src_all[test, 1]
def pick(mask, n=24):
    ids = np.where(mask)[0]; ids = ids[np.argsort(y_test[ids])]
    return ids[np.linspace(0, len(ids) - 1, n).round().astype(int)]
qpos = pick(y_test > 0); qneg = pick(y_test < 0)
print(f'queries: {len(qpos)} at +y (y {y_test[qpos].min():.1f}..{y_test[qpos].max():.1f}), {len(qneg)} at -y (y {y_test[qneg].min():.1f}..{y_test[qneg].max():.1f})')

def fwd(model, depth_coord, x, src_loc, ref_ir_locs, tgt_wav):
    """xRIR.forward, returning the per-reference per-time mixing weights as well."""
    x = model.shift_and_align(x, src_loc, ref_ir_locs)
    times = model.times
    time_embed = model.time_embedder(times.unsqueeze(0)).repeat(ref_ir_locs.shape[0], 1, 1)
    time_out = model.time_proj(time_embed)
    source_out = model.lin_proj_0(model.src_proj(model.source_network((src_loc[:, :, None, None] - depth_coord) / 5.)).permute(0, 2, 1)).squeeze(-1).unsqueeze(1)
    receiver_out = model.lin_proj_0(model.src_proj(model.source_network((-depth_coord) / 5.)).permute(0, 2, 1)).squeeze(-1).unsqueeze(1)
    ref_geo = torch.cat([model.lin_proj_0(model.src_proj(model.source_network((ref_ir_locs[:, i, :, None, None] - depth_coord) / 5.)).permute(0, 2, 1)).squeeze(-1).unsqueeze(1) for i in range(ref_ir_locs.shape[1])], 1)
    fuse_geo = torch.cat([receiver_out, source_out], -1)
    fuse_ref_geo = torch.cat([receiver_out.repeat(1, ref_geo.shape[1], 1), ref_geo], -1)
    ref_src = torch.cat([model.src_coord_proj(model.dist_embedder(ref_ir_locs[:, i:i + 1] / 5.).view(ref_ir_locs.shape[0], -1)).unsqueeze(1) for i in range(ref_ir_locs.shape[1])], 1)
    src_feats = model.src_coord_proj(model.dist_embedder(src_loc.unsqueeze(1) / 5.).view(src_loc.shape[0], -1)).unsqueeze(1)
    fuse_geo = torch.cat([src_feats, fuse_geo], -1); fuse_ref_geo = torch.cat([ref_src, fuse_ref_geo], -1)
    specs, logs, afe = [], [], []
    for i in range(x.shape[1]):
        s = model.convert_ir_to_spec(x[:, i:i + 1]); specs.append(s); logs.append(torch.log(s + 1e-8)); afe.append(model.audio_enc(s).unsqueeze(1))
    logs = torch.cat(logs, 1); afe = torch.cat(afe, 1)
    fuse_ref = model.lin_proj_2(torch.cat((fuse_ref_geo, afe), -1)); fuse_tgt = model.lin_proj_1(fuse_geo)
    fuse = F.softmax(fuse_ref @ fuse_tgt.permute(0, 2, 1) / fuse_tgt.shape[-1], dim=1) * fuse_ref
    weights = fuse @ time_out.permute(0, 2, 1) / fuse.shape[-1]                      # B N T
    out_log = torch.sum(logs * weights.unsqueeze(2), 1)                              # B 63 T
    return out_log, model.convert_ir_to_spec(tgt_wav)[:, 0], weights, source_out.squeeze(1)

def spec_c50(mag, onset_frames):
    e = (mag ** 2).sum(1)                                                            # B T
    out = []
    for b in range(e.shape[0]):
        o = int(onset_frames[b]); n50 = round(0.05 * 22050 / 31)
        out.append(10 * torch.log10(e[b, o:o + n50].sum() / (e[b, o + n50:].sum() + 1e-12)))
    return torch.stack(out)

def geo_feat(model, src_locs, depth_coord):
    with torch.no_grad():
        enc = (src_locs[:, :, None, None] - depth_coord[None]) / 5.
        return model.lin_proj_0(model.src_proj(model.source_network(enc)).permute(0, 2, 1)).squeeze(-1)

for name, backbone, ck in [('cyl', 'cylindrical', 'ckpt/xRIR_cyl_8_shot/epoch_12.pth'), ('control', 'simple', 'ckpt/xRIR_simple_8_shot/epoch_12.pth'), ('released', 'simple', 'checkpoints/xRIR_unseen.pth')]:
    model = build_xrir(backbone, 8).eval()
    sd = torch.load(ck, map_location='cpu'); sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    # (A) geometry-branch feature: +y query vs its mirror, vs nearest -y train mic, vs a +y neighbour 1.2 m away
    dc = D['depth_coord']
    with torch.no_grad():
        q = D['src_local'][test[qpos]]; mirror = q * torch.tensor([-1., -1., 1.])
        tr = D['src_local'][train]; d_tr = torch.cdist(q, tr); 
        near_neg = tr[[int(j) for j in [torch.argmin(torch.where(tr[:, 1] < 0, d_tr[b], torch.inf)) for b in range(len(q))]]]
        neighbour = q + torch.tensor([0., 1.2, 0.]) * torch.where(q[:, 1:2] < q[:, 1:2].median(), 1., -1.)
        fq, fm, fn, fb = [geo_feat(model, p, dc) for p in (q, mirror, near_neg, neighbour)]
        cos = lambda a, b: F.cosine_similarity(a, b, dim=-1).mean().item()
        rel = lambda a, b: ((a - b).norm(dim=-1) / a.norm(dim=-1)).mean().item()
        print(f'\n[{name}] geometry feature of a +y mic vs: exact mirror (-x,-y)  cos {cos(fq, fm):.4f} relL2 {rel(fq, fm):.3f} | nearest -y train mic cos {cos(fq, fn):.4f} relL2 {rel(fq, fn):.3f} | same-side neighbour 1.2 m cos {cos(fq, fb):.4f} relL2 {rel(fq, fb):.3f}')
    # (B) full forward: reference-weight share on -y references, and signed spectral C50 error, by query side
    for side, qs in [('+y', qpos), ('-y', qneg)]:
        shares, errs, ok = [], [], 0
        for b0 in range(0, len(qs), 8):
            batch = [ds[int(i)] for i in qs[b0:b0 + 8]]
            lp, sl, dcb, tw, ri, rl = [torch.stack([t[k] for t in batch]) for k in range(6)]
            with torch.no_grad():
                out_log, tgt, w, _ = fwd(model, dcb, ri, sl, rl, tw)
            neg = (rl[:, :, 1] < 0).float()                                          # B N
            aw = w.abs().mean(-1)                                                    # B N
            shares.append(((aw * neg).sum(1) / aw.sum(1)).numpy())
            onset = (sl.norm(dim=1) / 343. * 22050 / 31).round().long()
            errs.append((spec_c50(torch.exp(out_log), onset) - spec_c50(tgt, onset)).numpy())
            ok += (neg.mean(1) > 0).sum().item()
        shares = np.concatenate(shares); errs = np.concatenate(errs)
        print(f'[{name}] queries at {side}: share of |mixing weight| on -y references {shares.mean():.3f} (refs drawn: {neg.mean().item():.2f} are -y) | signed spectral-C50 error (pred - true) {errs.mean():+.2f} dB (sd {errs.std():.2f})')
