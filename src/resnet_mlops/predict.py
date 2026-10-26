"""Single-image inference shared by the CLI and the serving app."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from resnet_mlops.config import resolve_device
from resnet_mlops.data import CIFAR_MEAN, CIFAR_STD
from resnet_mlops.evaluate import load_checkpoint

_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.classes: list[str] = ckpt.get("classes") or [str(i) for i in range(ckpt["model_config"]["num_classes"])]

    @torch.no_grad()
    def predict_image(self, image: Image.Image, top_k: int = 5) -> dict:
        x = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1).squeeze(0).cpu()
        top = probs.topk(min(top_k, probs.numel()))
        return {
            "predictions": [
                {"class": self.classes[i], "probability": round(float(p), 6)}
                for p, i in zip(top.values.tolist(), top.indices.tolist())
            ]
        }

    def predict_bytes(self, payload: bytes, top_k: int = 5) -> dict:
        return self.predict_image(Image.open(BytesIO(payload)), top_k=top_k)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Classify an image with a trained checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    result = Predictor(args.checkpoint).predict_image(Image.open(args.image), top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
