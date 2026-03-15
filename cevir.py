import tensorflow as tf
model = tf.keras.models.load_model('kanser_tespit_modeli.keras')
model.save('kanser_tespit_modeli.h5')
print("Model başarıyla .h5 formatına çevrildi!")