from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from rembg import remove
import io

USE_REMBG = True # test both True and False 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("olive_multiclass_v3.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["Healthy", "aculus_olearius", "olive_peacock_spot"]  # must match training order

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "Olive Leaf Disease Classifier API (3 classes) is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if USE_REMBG:
        no_bg = remove(image)
        # white background, since the new dataset has light backgrounds
        bg = Image.new("RGB", no_bg.size, (255, 255, 255))
        bg.paste(no_bg, mask=no_bg.split()[3])
        image = bg
    image.save("debug_input.png")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
        idx = torch.argmax(probs).item()

    return {
        "prediction": class_names[idx],
        "confidence": round(probs[idx].item() * 100, 2),
        "probabilities": {class_names[i]: round(probs[i].item() * 100, 2) for i in range(3)},
    }