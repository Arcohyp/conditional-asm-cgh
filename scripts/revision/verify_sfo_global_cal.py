import sys, torch, numpy as np
from PIL import Image
import torchvision.transforms as T
SFO_DIR = '/mnt/sda/chengxirun/hologram_task/eghp_experiment/CGH-SFO-solver'
sys.path.insert(0, SFO_DIR)
from model.FourierNet import Flex_FourierNet
from prop import ASM_split

gpu = int(sys.argv[1]); color = sys.argv[2]
torch.cuda.set_device(gpu); device = torch.device(f'cuda:{gpu}')
RES = (2160, 3840); PITCH = 3.74e-3
wl_of = {'r': 638, 'g': 520, 'b': 450}; color_idx = {'r':0,'g':1,'b':2}
wl = wl_of[color]
prop = ASM_split(wavelength=wl, res=RES, roi=RES, pitch=PITCH, apply_constraint=True).to(device).eval()
model = Flex_FourierNet(wl=wl, center_distance=100).to(device)
model.load_state_dict(torch.load(f'{SFO_DIR}/model/{color}/FourierNet_flex_100.pth', map_location=device))
model.eval()
tfm = T.Compose([T.Resize(RES), T.ToTensor()])
import os
vdir = '/mnt/sda/datasets/DIV2K/DIV2K_valid_HR'
files = sorted(os.listdir(vdir))[:20]

recs, tgts = [], []
with torch.no_grad():
    for f in files:
        t = tfm(Image.open(os.path.join(vdir, f)).convert('RGB'))[color_idx[color]].to(device)
        target = t.unsqueeze(0).unsqueeze(0)
        dt = torch.tensor([[100.0]], device=device)
        holo = model(target, dt)
        _, _, final = prop(torch.ones_like(holo), holo, dt)
        recs.append(final.sqrt())
        tgts.append(target.sqrt())

# least-squares optimal SINGLE global amplitude constant: g* = sum<ra,ta>/sum<ra,ra>
num = sum(float((ra*ta).sum()) for ra, ta in zip(recs, tgts))
den = sum(float((ra*ra).sum()) for ra in recs)
g = num/den
print(f'{color}: LS-optimal global amplitude constant g*={g:.4f} (1/g*={1/g:.3f})')

for name, scale in [('raw (no scaling)', 1.0), ('one global constant g*', g)]:
    ps = []
    for ra, ta in zip(recs, tgts):
        mse = ((ra*scale - ta)**2).mean().item()
        ps.append(10*np.log10(1.0/mse))
    print(f'{color} z=100 fixed-peak amp PSNR, {name}: {np.mean(ps):.2f} dB')
