import torch
from collections import OrderedDict

config = 'waymo_D1_2x_3class_centerhead_iou_nms_2f'

ckpt = torch.load(f'./checkpoints/{config}/latest.pth')
model = ckpt['state_dict']
new_model = OrderedDict()
for name in model:
    if name.startswith("backbone"):
        new_name = name.replace("encoder_", "block.").replace("win_attn", "attn").replace("self_attn", "attn").replace("linear", "fc")
        new_model[new_name] = model[name]
    else:
        new_model[name] = model[name]
ckpt['state_dict'] = new_model
torch.save(ckpt, f'./checkpoints/{config}/converted.pth')