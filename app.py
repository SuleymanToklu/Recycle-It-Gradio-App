import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. Model ve Sınıf İsimlerini Yükle ---

# Eğittiğimiz modeli yükleyelim. 
# Hata kontrolü eklemek her zaman iyidir.
try:
    model = tf.keras.models.load_model('recycle_model.h5')
    print("Model 'recycle_model.h5' başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Model yüklenemedi. {e}")
    model = None

# Modelin tahmin edeceği sayısal etiketlere karşılık gelen sınıf isimleri.
# Bu listenin sırası, modelin eğitimi sırasında belirlenen sırayla aynı olmalıdır.
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print("Sınıf isimleri tanımlandı:", class_names)


# --- 2. Tahmin Fonksiyonunu Tanımla ---
# Bu fonksiyon, Gradio arayüzünden bir resim alacak ve bir tahmin döndürecek.
def predict(input_image: Image.Image):
    if model is None or input_image is None:
        return "Model yüklenemedi veya resim sağlanmadı."

    # Gelen PIL resmini modelin anlayacağı formata dönüştür
    # 1. Yeniden boyutlandırma (eğitimde kullandığımız boyutla aynı olmalı)
    img = input_image.resize((224, 224))
    
    # 2. NumPy array'e çevirme
    img_array = np.array(img)
    
    # 3. Batch boyutu ekleme (model tek resim yerine bir grup resim bekler)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. MobileNetV2'nin beklediği ön işlemeyi uygulama (-1 ile 1 arasına ölçekleme)
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Tahmin yap
    prediction = model.predict(preprocessed_img)
    
    # Tahmin sonucunu, sınıf isimleri ve olasılıkları içeren bir dictionary'ye dönüştür
    # Bu format, Gradio'nun Label bileşeni için en uygun formattır.
    confidences = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    
    return confidences


# --- 3. Gradio Arayüzünü Oluştur ve Başlat ---

# Arayüz başlığı, açıklaması ve örnek resimler
title = "Recycle-It! ♻️ - Atık Sınıflandırma Uygulaması"
description = """
Bu uygulama, yüklediğiniz bir atık resminin hangi materyal olduğunu tahmin eder. 
Fotoğrafınızı yükleyin veya aşağıdaki örneklerden birini deneyin. 
Model, Transfer Öğrenmesi (MobileNetV2) ile eğitilmiş bir Evrişimli Sinir Ağıdır.
"""
examples = [
    'data/garbage_classification/garbage_classification/paper/paper1.jpg',
    'data/garbage_classification/garbage_classification/glass/glass1.jpg',
    'data/garbage_classification/garbage_classification/plastic/plastic1.jpg'
]

# Arayüzü inşa et
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Atık Fotoğrafı Yükle"),
    outputs=gr.Label(num_top_classes=3, label="Tahmin Sonuçları"),
    title=title,
    description=description,
    examples=examples
)

# Uygulamayı başlat
iface.launch()