import sys
from pathlib import Path

# Ampiana eto ny lalana mankany amin'ny root (project root) mba ho hitan'i Python i 'model'
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np

print('Python:', sys.version.split()[0])

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print('TensorFlow: NOT INSTALLED')
    print('Source/build files are valid, but model validation requires: pip install -r requirements.txt')
    raise SystemExit(2)

from model import build_model

print('TensorFlow:', tf.__version__)
model = build_model(128, 12)
x = np.zeros((1, 5, 9, 7), dtype=np.float32)
policy, value = model(x, training=False)
assert tuple(policy.shape) == (1, 1080), policy.shape
assert tuple(value.shape) == (1, 1), value.shape
print('Input: ', x.shape)
print('Policy:', policy.shape)
print('Value: ', value.shape)
print('MODEL BUILD CHECK PASSED')