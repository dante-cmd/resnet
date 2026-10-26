VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup train evaluate serve test docker-build smoke

setup:
	python3 -m venv $(VENV)
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

smoke:      ## 1-epoch CPU training run to verify the pipeline end to end
	$(PY) -m resnet_mlops.train --config configs/train.yaml --set train.epochs=1 data.batch_size=64

train:      ## full training run (config in configs/train.yaml)
	$(PY) -m resnet_mlops.train --config configs/train.yaml

evaluate:   ## evaluate checkpoints/best.pt on the test set
	$(PY) -m resnet_mlops.evaluate --config configs/train.yaml --checkpoint checkpoints/best.pt

serve:      ## run the inference API locally
	MODEL_CHECKPOINT=checkpoints/best.pt $(VENV)/bin/uvicorn serving.app:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

docker-build:  ## build the serving image with the trained checkpoint baked in
	docker build -f serving/Dockerfile -t resnet-mlops --build-arg CHECKPOINT=checkpoints/best.pt .

mlflow-ui:  ## inspect runs locally
	$(VENV)/bin/mlflow ui --backend-store-uri ./mlruns
