import gradio as gr
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

# --- 1. Model ve Sınıf İsimlerini Yükle ---

# PyTorch ile eğitilmiş modeli yükle
model = None
model_path = 'recycle_model.pt'
if os.path.exists(model_path):
    try:
        from torchvision import models
        model = models.mobilenet_v2(weights=None)
        num_classes = 6
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model '{model_path}' başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: Model yüklenemedi. {e}")
        model = None
else:
    print(f"Model dosyası bulunamadı: {model_path}")

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print("Sınıf isimleri tanımlandı:", class_names)

# --- 2. Tahmin Fonksiyonunu Tanımla ---
def predict(input_image: Image.Image):
    if model is None:
        return "Model yüklenemedi."
    if input_image is None:
        return "Lütfen bir resim yükleyin veya kameradan çekin."

    # PyTorch için ön işleme
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(input_image).unsqueeze(0)  # batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).numpy()

    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return confidences

# --- 3. Gradio Arayüzünü Oluştur ve Başlat ---
title = "Recycle-It! ♻️ - Atık Sınıflandırma Uygulaması"
description = """
Bu uygulama, yüklediğiniz bir atık resminin hangi materyal olduğunu tahmin eder.\n\nFotoğrafınızı yükleyin, kameradan çekin veya aşağıdaki örneklerden birini deneyin.\nModel, Transfer Öğrenmesi (MobileNetV2) ile eğitilmiş bir Evrişimli Sinir Ağıdır.
"""
examples = [
    'data/paper/paper1.jpg',
    'data/glass/glass1.jpg',
    'data/plastic/plastic1.jpg'
]

# Hakkında sekmesi için içerik
about_md = """
# Recycle-It! ♻️\n\n"""
about_md += "Bu uygulama, atıkların doğru şekilde ayrıştırılmasına yardımcı olmak için geliştirilmiştir.\n"
about_md += "Bir atık fotoğrafı yükleyerek veya kameradan çekerek, atığın hangi kategoriye ait olduğunu (karton, cam, metal, kağıt, plastik, çöp) hızlıca öğrenebilirsiniz.\n\n"
about_md += "**Nasıl Kullanılır?**\n"
about_md += "- 'Tahmin' sekmesine geçin.\n"
about_md += "- Fotoğraf yükleyin veya kameradan çekin.\n"
about_md += "- Sonuçlar ekranda görüntülenecektir.\n\n"
about_md += "Model, MobileNetV2 mimarisiyle eğitilmiştir ve 6 farklı atık türünü ayırt edebilir.\n"

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    with gr.Tab("Hakkında"):
        gr.Markdown(about_md)
    with gr.Tab("Tahmin"):
        gr.Markdown(description)
        gr.Interface(
            fn=predict,
            inputs=gr.Image(type="pil", label="Atık Fotoğrafı Yükle veya Kameradan Çek", sources=["upload", "webcam"]),
            outputs=gr.Label(num_top_classes=3, label="Tahmin Sonuçları"),
            examples=examples,
            allow_flagging="never"
        )

demo.launch()