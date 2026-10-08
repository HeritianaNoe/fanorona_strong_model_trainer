# Fanorona Strong AI Trainer

Python trainer for the Flutter Fanorona program.

## Model contract

The generated model is compatible with `NNEncoding` in the Flutter app:

- input: `[1, 5, 9, 7]` float32
- policy: `[1, 1080]` float32 logits
- value: `[1, 1]` float32 in `[-1, 1]`
- action id: `((row*9 + col)*8 + directionIndex)*3 + moveType`
- directions: N, S, W, E, NW, NE, SW, SE
- move types: normal=0, approach=1, withdrawal=2

The trainer implements the same 5x9 board, strong-point diagonals, forced capture,
approach/withdrawal capture, and Riatra capture chains used by the Dart source.

## Recommended strong training

1. Install Python 3.11.
2. `py -3.11 -m venv .venv`
3. `.venv\\Scripts\\activate`
4. `pip install -r requirements.txt`
5. Start a first run:
   `python train_fanorona.py --iterations 12 --games-per-iteration 80 --simulations 600 --epochs 4`
6. For a much stronger model, continue:
   `python train_fanorona.py --iterations 40 --games-per-iteration 300 --simulations 1600 --epochs 6 --resume checkpoints/latest.keras`
7. Export:
   `python convert_tflite.py --checkpoint checkpoints/latest.keras --output fanorona_model.tflite`

For a serious CPU training run, increase games and simulations rather than only increasing
network size. The self-play data is the important part.

## Fast test

`python train_fanorona.py --iterations 1 --games-per-iteration 4 --simulations 80 --epochs 1`

Then:
`python convert_tflite.py --checkpoint checkpoints/latest.keras --output fanorona_model.tflite`
"# fanorona_strong_model_trainer" 
