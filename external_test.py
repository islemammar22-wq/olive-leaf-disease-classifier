import os, random
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from rembg import remove

BASE = r"C:\Users\islem\Downloads\Zeytin_224x224_Augmented\test"
FOLDERS = {
    os.path.join(BASE, "Saglikli"): "Healthy",
    os.path.join(BASE, "Pas_Akari"): "aculus_olearius",
    os.path.join(BASE, "Kus_Gozu_Mantari"): "olive_peacock_spot",
}
MAX_PER_FOLDER = 30
class_names = ["Healthy", "aculus_olearius", "olive_peacock_spot"]

device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("olive_multiclass_v3.pth", map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(img, use_rembg):
    if use_rembg:
        nb = remove(img)
        bg = Image.new("RGB", nb.size, (255, 255, 255))
        bg.paste(nb, mask=nb.split()[3])
        img = bg
    with torch.no_grad():
        return class_names[model(tf(img).unsqueeze(0)).argmax(1).item()]

# Same random sample for both passes
random.seed(0)
samples = {}
for folder in FOLDERS:
    all_files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    samples[folder] = random.sample(all_files, min(MAX_PER_FOLDER, len(all_files)))

for use_rembg in (False, True):
    correct = total = 0
    for folder, true_label in FOLDERS.items():
        ok = 0
        for f in samples[folder]:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            ok += predict(img, use_rembg) == true_label
        print(f"rembg={use_rembg} | {true_label}: {ok}/{len(samples[folder])}")
        correct += ok
        total += len(samples[folder])
    print(f"rembg={use_rembg} | TOTAL: {correct}/{total} = {100*correct/total:.1f}%\n")