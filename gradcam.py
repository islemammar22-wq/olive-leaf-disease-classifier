import sys, os
import numpy as np
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class_names = ["Healthy", "aculus_olearius", "olive_peacock_spot"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("olive_multiclass_v3.pth", map_location="cpu"))
model.eval()

store = {}
def fwd_hook(module, inp, out):
    store["a"] = out
    out.register_hook(lambda g: store.__setitem__("g", g))
model.layer4.register_forward_hook(fwd_hook)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

for path in sys.argv[1:]:
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)

    out = model(x)                      # no torch.no_grad(): we need gradients
    probs = torch.softmax(out, 1)[0]
    idx = probs.argmax().item()
    print(f"{os.path.basename(path)} -> {class_names[idx]} ({probs[idx]*100:.1f}%)")

    model.zero_grad()
    out[0, idx].backward()

    a, g = store["a"][0], store["g"][0]          # (512,7,7)
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((w * a).sum(0)).detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    heat = np.array(cam_img).astype(np.float32) / 255.0
    base = np.array(img).astype(np.float32)
    overlay = base.copy()
    overlay[..., 0] = np.clip(base[..., 0] + 255 * heat, 0, 255)   # red = high attention
    overlay[..., 1] = base[..., 1] * (1 - 0.6 * heat)
    overlay[..., 2] = base[..., 2] * (1 - 0.6 * heat)
    overlay = overlay.astype(np.uint8)

    side = np.concatenate([base.astype(np.uint8), overlay], axis=1)
    out_name = f"gradcam_{os.path.splitext(os.path.basename(path))[0]}.png"
    Image.fromarray(side).save(out_name)
    print("saved", out_name)