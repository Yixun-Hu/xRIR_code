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
def pick(mask, n=16):
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


tr_side = np.sign(src_all[train, 1])
def refs_for(side_sign):
    return torch.as_tensor(train[tr_side == side_sign], dtype=torch.long)
print('same-side reference pools: +y', int((tr_side > 0).sum()), 'mics, -y', int((tr_side < 0).sum()), 'mics')
for name, backbone, ck in [('cyl', 'cylindrical', 'ckpt/xRIR_cyl_8_shot/epoch_12.pth'), ('control', 'simple', 'ckpt/xRIR_simple_8_shot/epoch_12.pth'), ('released', 'simple', 'checkpoints/xRIR_unseen.pth')]:
    model = build_xrir(backbone, 8).eval()
    sd = torch.load(ck, map_location='cpu'); sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    for side, qs, sgn in [('+y', qpos, 1), ('-y', qneg, -1)]:
        for cond in ['standard K=8', 'same-side only', 'other-side only']:
            c50e, l1 = [], []
            for b0 in range(0, len(qs), 8):
                ids = [int(test[i]) for i in qs[b0:b0 + 8]]
                sl = D['src_local'][ids]; tw = D['rirs'][ids].unsqueeze(1); dcb = D['depth_coord'][None].expand(len(ids), -1, -1, -1)
                if cond == 'standard K=8':
                    refs = [torch.as_tensor(ds._pick_refs('hallway', i), dtype=torch.long) for i in ids]
                else:
                    r = refs_for(sgn if cond == 'same-side only' else -sgn); refs = [r] * len(ids)
                ri = torch.stack([D['rirs'][r] for r in refs]); rl = torch.stack([D['src_local'][r] for r in refs])
                with torch.no_grad():
                    out_log, tgt, w, _ = fwd(model, dcb, ri, sl, rl, tw)
                onset = (sl.norm(dim=1) / 343. * 22050 / 31).round().long()
                c50e.append((spec_c50(torch.exp(out_log), onset) - spec_c50(tgt, onset)).abs().numpy())
                l1.append((out_log - torch.log(tgt + 1e-8)).abs().mean((1, 2)).numpy())
            c50e = np.concatenate(c50e); l1 = np.concatenate(l1)
            print(f'[{name:8s}] queries {side} | {cond:16s} | spectral C50 abs err {c50e.mean():5.2f} dB | log-spec L1 {l1.mean():.3f}')
