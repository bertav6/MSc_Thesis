import tensorflow as tf
import keras

# Load model
model_path = '/Users/bertavinas/Documents/DTU/Thesis/Code/code_submission/models/'
model_type = 'DSCNN'
model_name = 'retrained_9BC_50epochs'

model = keras.models.load_model(model_path+model_name+'.h5')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(model_name+'.tflite', 'wb') as f:
  f.write(tflite_model)

