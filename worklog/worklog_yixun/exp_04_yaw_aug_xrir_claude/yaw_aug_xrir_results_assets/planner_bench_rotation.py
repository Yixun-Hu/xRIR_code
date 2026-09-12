import torch, time, statistics, sys
sys.path.insert(0, "/home/yixunhu/codespace/xRIR_code")
from tools.yaw_aug import apply_yaw_aug, YawAug
torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=1 -> device 0 here
B = 32
d = torch.randn(B, 3, 256, 512, device="cuda"); s = torch.randn(B, 3, device="cuda"); r = torch.randn(B, 8, 3, device="cuda")
aug = YawAug(True, 512, 0)
def one(i):
    ks = aug.offsets_for(1, i, B)
    return apply_yaw_aug(d, s, r, ks.to(d.device))
for i in range(10): one(i)
torch.cuda.synchronize()
ts = []
for i in range(100):
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record(); one(i); en.record(); torch.cuda.synchronize(); ts.append(st.elapsed_time(en))
wall = []
for i in range(100):
    torch.cuda.synchronize(); t0 = time.perf_counter(); one(i); torch.cuda.synchronize(); wall.append((time.perf_counter() - t0) * 1000)
print("apply_yaw_aug B=32 [3,256,512]: cuda-event ms min %.3f median %.3f p90 %.3f | wall ms min %.3f median %.3f" % (
    min(ts), statistics.median(ts), sorted(ts)[89], min(wall), statistics.median(wall)))
print("bound vs 1.4 s/iter dedicated: min %.2f%% median %.2f%%" % (100*min(wall)/1400, 100*statistics.median(wall)/1400))
