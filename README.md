# resnet-mlops

An MLOps-style ResNet (He et al., 2015, implemented from scratch in PyTorch) covering the full
lifecycle: **train → validate → test → track → deploy**.

```
configs/train.yaml        single source of truth for hyperparameters
src/resnet_mlops/
  config.py               typed config (dataclasses) + YAML + CLI overrides
  model.py                ResNet-18/50 from scratch (BasicBlock / Bottleneck)
  data.py                 CIFAR-10/100 or ImageFolder; stratified train/val split
  train.py                train loop, MLflow tracking, best/last checkpoints, early stopping
  evaluate.py             test-set evaluation: accuracy, precision/recall/F1, confusion matrix
  predict.py              single-image inference (shared by CLI + API)
serving/
  app.py                  FastAPI service (/health /ready /predict /predict/url)
  Dockerfile              CPU-slim image with the checkpoint baked in
tests/                    unit tests (model, config, checkpoint round-trip)
Makefile                  setup / smoke / train / evaluate / serve / test / docker-build
```

## Quick start

```bash
make setup        # venv + CPU torch + deps (no GPU assumed)
make smoke        # 1-epoch run to verify the whole pipeline
make train        # full training run, tracked in MLflow (./mlruns)
make evaluate     # test-set metrics for checkpoints/best.pt
make serve        # inference API on :8000
make test
```

Individual steps without make:

```bash
python -m resnet_mlops.train  --config configs/train.yaml --set train.epochs=5
python -m resnet_mlops.evaluate --checkpoint checkpoints/best.pt
python -m resnet_mlops.predict  --checkpoint checkpoints/best.pt --image cat.png
uvicorn serving.app:app --port 8000
```

## MLOps practices baked in

- **Experiment tracking** — every run logs params, per-epoch metrics, best epoch,
  training time and the best checkpoint artifact to MLflow (`make mlflow-ui` to browse).
- **Reproducibility** — fixed seed, deterministic data split, all hyperparameters in
  versioned YAML, model config stored inside every checkpoint.
- **Clean data separation** — stratified 90/10 train/validation split for model
  selection; the test set is only touched by `evaluate.py`.
- **Model selection** — best checkpoint chosen by validation accuracy, with early
  stopping; `last.pt` is kept for crash recovery.
- **Evaluation as a separate stage** — accuracy / macro precision / recall / F1,
  confusion matrix and full classification report written to `outputs/`.
- **Packaging & deployment** — the package installs via `pyproject.toml`; the
  FastAPI service ships as a slim CPU Docker image with the model baked in:
  `docker run -p 8000:8000 resnet-mlops`.

## Deploying

```bash
docker build -f serving/Dockerfile -t resnet-mlops --build-arg CHECKPOINT=checkpoints/best.pt .
docker run -p 8000:8000 resnet-mlops
curl -F "file=@cat.png" http://localhost:8000/predict
```

Point `tracking.tracking_uri` in `configs/train.yaml` at an MLflow server to share
experiments across a team instead of the local `./mlruns` store.

## Switching datasets / model depth

In `configs/train.yaml`:

- `data.dataset: imagefolder` with `data.root` pointing at `root/train/<class>/...`
  and `root/test/<class>/...` (ImageFolder layout).
- `model.blocks: [3, 4, 6, 3]` for a ResNet-50 (Bottleneck).
