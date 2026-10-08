import sys
import numpy as np
import tensorflow as tf

if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/validate_tflite.py fanorona_model.tflite")

path = sys.argv[1]
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()

print("Input details:", inputs)
print("Output details:", outputs)

assert len(inputs) == 1, "Expected one input tensor"
assert tuple(inputs[0]["shape"]) == (1, 5, 9, 7), inputs[0]["shape"]
assert len(outputs) == 2, "Expected policy + value outputs"

interpreter.set_tensor(inputs[0]["index"], np.zeros((1, 5, 9, 7), np.float32))
interpreter.invoke()
vals = [interpreter.get_tensor(o["index"]) for o in outputs]
shapes = [tuple(v.shape) for v in vals]
print("Output shapes:", shapes)
assert (1, 1080) in shapes, shapes
assert (1, 1) in shapes, shapes
print("TFLITE VALIDATION PASSED")