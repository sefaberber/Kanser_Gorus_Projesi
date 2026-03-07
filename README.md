# 🧠 NeuroAI Vision

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=for-the-badge&logo=flask)
![HTML5](https://img.shields.io/badge/HTML5-UI-e34f26?style=for-the-badge&logo=html5)

> **Yapay Sinir Ağları (CNN) kullanılarak geliştirilmiş, web tabanlı bir beyin tümörü tespit ve MR analiz aracı.**

NeuroAI Vision, yüklenen beyin MR görüntülerini saniyeler içinde analiz ederek olası tümör oluşumlarını tespit eden bir derin öğrenme projesidir. Tıbbi görüntü işleme alanındaki karmaşık yapay zeka süreçlerini, herkesin kullanabileceği fütüristik ve modern bir web arayüzü ile birleştirir. *(Önemli Not: Bu uygulama makine öğrenmesi teknolojilerini sergilemek amacıyla tasarlanmış bir eğitim prototipidir ve hiçbir koşulda gerçek tıbbi teşhis veya klinik karar verme süreçlerinde kullanılamaz.)*

## ✨ Öne Çıkan Özellikler

* **🔬 Derin Öğrenme Mimarisi:** TensorFlow ve Keras altyapısı ile sıfırdan eğitilmiş Convolutional Neural Network (CNN) modeli.
* **🎯 Üçlü Sınıflandırma Sistemi:** Model, MR kesitlerini 3 farklı tıbbi kategoride yüksek doğrulukla analiz eder: `Normal` (Sağlıklı Beyin Dokusu), `Meningioma` (İyi Huylu Tümör), `Glioma` (Kötü Huylu Tümör).
* **🌌 Modern "Glassmorphism" Arayüz:** Flask backend'i ile entegre çalışan; karanlık tema (Dark Mode), cam efekti ve sonuca göre değişen dinamik neon animasyonlara sahip fütüristik UI tasarımı.
* **⚡ Bulut Optimizasyonu:** Sunucu tarafında minimum kaynak tüketimi için `tensorflow-cpu` ile entegre edilmiş, hafif ve hızlı çalışan backend yapısı.

## 🧠 Model Anatomisi

Sistem, yaklaşık 2000 adet etiketli MR görseliyle eğitilmiştir. Görüntüler işlem öncesi `224x224` piksel boyutuna standardize edilir ve piksel değerleri `[0, 1]` aralığında normalize edilir. Ağ mimarisi 3 adet *Conv2D + MaxPooling2D* bloğundan ve ardından gelen *Dense* katmanlarından oluşur. Çıkış katmanı, 3 sınıf için `softmax` aktivasyon fonksiyonunu kullanır.

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi yerel ortamınızda (localhost) çalıştırmak ve test etmek için projenin ana dizininde terminali açıp aşağıdaki komutları sırasıyla girmeniz yeterlidir:

```bash
pip install -r requirements.txt
python app.py

## Sunucu başlatıldıktan sonra tarayıcınızdan http://127.0.0.1:5000 adresine giderek sistemi hemen kullanmaya başlayabilirsiniz.