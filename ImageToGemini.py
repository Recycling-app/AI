import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import google.generativeai as genai

genai.configure(api_key="APIKEY") # API키
gemini_model = genai.GenerativeModel("gemini-2.5-flash") # 모델 선택

class_names = ['건전지', '금속', '비닐', '스티로폼', '유리', '종이', '종이박스', '플라스틱', '형광등']

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL_DIR = "C:\\Users\\myong\\Desktop\\final_model"
MODEL_FILES = {
    "Train_model8": os.path.join(MODEL_DIR, "Train_model8.ptl"),
    "Train_model9": os.path.join(MODEL_DIR, "Train_model9.ptl"),
    "Train_model13": os.path.join(MODEL_DIR, "Train_model13.ptl")
}

loaded_models = {
    name: torch.jit.load(path).eval()
    for name, path in MODEL_FILES.items()
    if os.path.exists(path)
}
        
model_8 = loaded_models["Train_model8"]
model_9 = loaded_models["Train_model9"]
model_10 = loaded_models["Train_model13"]

test_folder = "C:\\Users\\myong\\Desktop\\test"

for filename in os.listdir(test_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(test_folder, filename)
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                prob_8 = F.softmax(model_8(input_tensor), dim=1)
                prob_9 = F.softmax(model_9(input_tensor), dim=1)
                prob_10 = F.softmax(model_10(input_tensor), dim=1)
                avg_prob = (prob_8 + prob_9 + prob_10) / 3

                final_prediction_index = avg_prob.argmax(dim=1).item()
                predicted_label = class_names[final_prediction_index]
                confidence = avg_prob.max().item() * 100

            print(f"\n========================================")
            print(f"파일명: {filename}")
            print(f"예측 결과: {predicted_label} (신뢰도: {confidence:.2f}%)")
            
            # Gemini에 요청
            prompt = f"'{predicted_label}' 에 대한 분리수거 방법을 간단하게 알려줘"
            response = gemini_model.generate_content(prompt)

            print("Gemini 분리수거 방법")
            print(response.text.strip())
            print("========================================")

        except Exception as e:
            print(f"\n예측 중 오류 발생 ({filename}): {e}")