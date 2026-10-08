import argparse, tensorflow as tf

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', default='fanorona_model.tflite')
a = p.parse_args()

model = tf.keras.models.load_model(a.checkpoint)

# Keep float32: the Flutter encoder and current NNEncoding expect float tensors and this avoids
# quantization changing policy/value calibration during the first strong-model tests.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
tflite = converter.convert()

open(a.output, 'wb').write(tflite)
print('saved', a.output, len(tflite), 'bytes')