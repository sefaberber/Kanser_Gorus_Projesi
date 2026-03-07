import tensorflow as tf
from tensorflow.keras import layers, models, Input

# 1. VERİ HAZIRLIĞI
train_dir = 'Training'
test_dir = 'Testing'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 

print("\n--- 1. Veriler Yükleniyor ---")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, shuffle=True, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

# 2. YAPAY SİNİR AĞI (CNN) MODELİNİN İNŞASI
model = models.Sequential([
    Input(shape=(224, 224, 3)), # Fotoğraf boyutunu algılama katmanı
    layers.Rescaling(1./255),   # Renkleri normalize etme
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 Sonuç: Normal, İyi, Kötü
])

# 3. MODELİ DERLEME
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. MODELİ EĞİTME (ÖĞRENME AŞAMASI)
print("\n--- 2. Eğitim Başlıyor (Bu işlem bilgisayarının hızına göre biraz sürebilir) ---")
# Yapay zeka tüm fotoğrafların üzerinden 10 kere (epoch) geçerek çalışacak
EPOCHS = 10 

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# 5. BEYNİ (MODELİ) KAYDETME
print("\n--- 3. Eğitim Tamamlandı! Model Kaydediliyor ---")
model.save('kanser_tespit_modeli.keras')
print("Harika! Öğrenilmiş model 'kanser_tespit_modeli.keras' adıyla klasörüne kaydedildi.")