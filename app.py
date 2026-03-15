from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename


# Web sunucusunu başlatıyoruz
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Yapay Zeka Modelimizi Yüklüyoruz
print("Yapay Zeka Modeli Web Sitesi İçin Hazırlanıyor...")
model = tf.keras.models.load_model('kanser_tespit_modeli.h5')
siniflar = ['Kötü Huylu Tümör (Glioma)', 'İyi Huylu Tümör (Meningioma)', 'Normal (Kanser Yok)']

# Ana Sayfa Yönlendirmesi
@app.route('/', methods=['GET', 'POST'])
def index():
    sonuc = None
    hata = None
    
    # Eğer butona basılıp fotoğraf gönderildiyse
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', hata='Dosya yüklenmedi.')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', hata='Herhangi bir dosya seçilmedi.')
        
        if file:
            # Fotoğrafı güvenli bir şekilde uploads klasörüne kaydet
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Kaydedilen fotoğrafı Yapay Zekaya ver
            img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            tahminler = model.predict(img_array)
            indeks = np.argmax(tahminler[0])
            
            # Sonucu HTML'e gönderilmek üzere paketle
            sonuc = {
                'sinif': siniflar[indeks],
                'oran': round(100 * np.max(tahminler[0]), 2)
            }
            
    return render_template('index.html', sonuc=sonuc, hata=hata)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)