import tensorflow as tf
import numpy as np

# 1. Eğittiğimiz Yapay Zeka Modelini Çağırıyoruz
print("\nModel yükleniyor, lütfen bekleyin...")
model = tf.keras.models.load_model('kanser_tespit_modeli.keras')

# 2. Test Edeceğimiz Fotoğrafın Yolu (BURAYI DEĞİŞTİRECEĞİZ)
resim_yolu = 'Testing/glioma_tumor/image(10).jpg' 

# 3. Fotoğrafı Yapay Zekanın Anlayacağı 224x224 Formata Çevirme
img = tf.keras.utils.load_img(resim_yolu, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Tek bir fotoğrafı işlenecek pakete dönüştürüyoruz

# 4. Yapay Zekadan Tahmin İsteme
tahminler = model.predict(img_array)

# 5. Sonucu Anlaşılır Şekilde Ekrana Yazdırma
# Sınıf sıramız klasör isimlerine göre: 0:glioma, 1:meningioma, 2:no_tumor
siniflar = ['Kötü Huylu Tümör (Glioma)', 'İyi Huylu Tümör (Meningioma)', 'Normal (Kanser Yok)']

en_yuksek_ihtimal_indeksi = np.argmax(tahminler[0])
tahmin_edilen_sinif = siniflar[en_yuksek_ihtimal_indeksi]
guven_orani = 100 * np.max(tahminler[0])

print(f"\n" + "="*40)
print(f"🧠 TAHMİN SONUCU")
print(f"="*40)
print(f"Yapay Zeka bu MR görüntüsünün %{guven_orani:.2f} ihtimalle")
print(f">>> {tahmin_edilen_sinif} <<<")
print(f"olduğunu tespit etti.")
print(f"="*40 + "\n")