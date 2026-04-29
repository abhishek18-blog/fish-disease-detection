from enum import Enum
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Smart Fish Disease Multi-Model Diagnosis API")

CLASSES = [
    'Bacterial Red disease', 
    'Bacterial diseases - Aeromoniasis', 
    'Bacterial gill disease', 
    'Fungal diseases Saprolegniasis', 
    'Healthy Fish', 
    'Parasitic diseases', 
    'Viral diseases White tail disease'
]

device = torch.device("cpu")

class ModelChoice(str, Enum):
    mobilenet = "MobileNetV2"
    resnet = "ResNet-50"
    densenet = "DenseNet-121"
    efficientnet = "EfficientNet-B0"

print("Loading AI Models into memory...")
loaded_models = {}

m_mobile = models.mobilenet_v2(weights=None)
m_mobile.classifier[1] = nn.Linear(m_mobile.classifier[1].in_features, len(CLASSES))
m_mobile.load_state_dict(torch.load('smart_fish_mobilenet.pth', map_location=device))
m_mobile.eval()
loaded_models[ModelChoice.mobilenet] = m_mobile

m_res = models.resnet50(weights=None)
m_res.fc = nn.Linear(m_res.fc.in_features, len(CLASSES))
m_res.load_state_dict(torch.load('smart_fish_resnet50.pth', map_location=device))
m_res.eval()
loaded_models[ModelChoice.resnet] = m_res

m_dense = models.densenet121(weights=None)
m_dense.classifier = nn.Linear(m_dense.classifier.in_features, len(CLASSES))
m_dense.load_state_dict(torch.load('smart_fish_densenet.pth', map_location=device))
m_dense.eval()
loaded_models[ModelChoice.densenet] = m_dense

m_eff = models.efficientnet_b0(weights=None)
m_eff.classifier[1] = nn.Linear(m_eff.classifier[1].in_features, len(CLASSES))
m_eff.load_state_dict(torch.load('smart_fish_efficientnet.pth', map_location=device))
m_eff.eval()
loaded_models[ModelChoice.efficientnet] = m_eff

print("All models loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Welcome to Smart Fish Diagnosis!"}

@app.post("/predict")
async def predict_disease(
    model_selection: ModelChoice = Form(..., description="Select the AI architecture to use"),
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        active_model = loaded_models[model_selection]
        
        with torch.no_grad():
            outputs = active_model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = CLASSES[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        
        return JSONResponse(content={
            "status": "success",
            "model_used": model_selection.value,
            "diagnosis": predicted_class,
            "confidence": f"{confidence_score:.2f}%"
        })
        
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)