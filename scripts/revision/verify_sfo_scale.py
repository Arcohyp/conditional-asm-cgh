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
files = sorted(os.listdir('/mnt/sda/datasets/DIV2K/DIV2K_valid_HR'))[:10]
scales, peakratios = [], []
with torch.no_grad():
    for f in files:
        t = tfm(Image.open(os.path.join('/mnt/sda/datasets/DIV2K/DIV2K_valid_HR', f)).convert('RGB'))[color_idx[color]].to(device)
        target = t.unsqueeze(0).unsqueeze(0)
        dt = torch.tensor([[100.0]], device=device)
        holo = model(target, dt)
        _, _, final = prop(torch.ones_like(holo), holo, dt)
        s = (target.sum()/final.sum()).item()
        scales.append(s)
        peakratios.append((final.max()/target.max()).item())
print(f'{color}: per-image sum-recovery scale (target.sum/final.sum): mean={np.mean(scales):.3f} min={np.min(scales):.3f} max={np.max(scales):.3f}')
print(f'{color}: peak ratio final/target: mean={np.mean(peakratios):.3f}')
