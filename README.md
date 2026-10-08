# Fanorona Strong AI Trainer

Python self-play trainer for the Fanorona Flutter AI project.

## Model contract

- Input: `[1, 5, 9, 7]` float32
- Policy: `[1, 1080]` float32 logits
- Value: `[1, 1]` float32 in `[-1, 1]`
- Action: `((row*9 + col)*8 + direction)*3 + moveType`

## Windows build

**Python 3.11 is recommended.** Run:

```bat
build.bat
```

This creates `.venv`, installs dependencies, compiles all Python sources, and validates the model contract.

## Quick training

```bat
train_strong.bat 1 4 80 1
```

Arguments:

`iterations games_per_iteration mcts_simulations epochs`

For stronger local training, for example:

```bat
train_strong.bat 40 300 1600 6
```

The exported file is `fanorona_model.tflite`.

## GitHub Actions

`Build and Validate` runs on push/PR.

`Train Fanorona Model` can be started manually from **Actions → Train Fanorona Model → Run workflow**. The resulting TFLite model is uploaded as a workflow artifact.

Do not commit generated checkpoints or TFLite binaries; they are ignored by `.gitignore`.
