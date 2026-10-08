import argparse, numpy as np, tensorflow as tf
from fanorona_engine import *
from model import ACTIONS

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--games', type=int, default=20)
a = p.parse_args()

model = tf.keras.models.load_model(a.model)
score = {BLACK: 0, WHITE: 0, 0: 0}

for g in range(a.games):
    s = State()
    for _ in range(260):
        turns = complete_turns(s)
        if not turns:
            break
        x = np.expand_dims(canonical_planes(s), 0)
        logits, val = model(x, training=False)
        logits = logits.numpy()[0]
        
        # Legal-action masking exactly as the Flutter side does.
        ids = [mv.action_id for mv, _ in turns]
        best = max(range(len(turns)), key=lambda i: logits[ids[i]])
        s = turns[best][1]
        
        if winner(s):
            break
            
    w = winner(s)
    score[w] += 1
    print('game', g + 1, 'winner', w)

print(score)