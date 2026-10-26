import pytest
import torch

from resnet_mlops.config import Config, _apply_overrides, load_config
from resnet_mlops.model import BasicBlock, Bottleneck, ResNet, resnet18, resnet50


def test_resnet18_output_shape():
    model = resnet18(num_classes=10)
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_resnet50_bottleneck_depth():
    model = resnet50(num_classes=100)
    assert isinstance(model.layer1[0], Bottleneck)
    assert model.fc.out_features == 100


def test_resnet_custom_blocks():
    model = ResNet(num_classes=5, blocks=[1, 1, 1, 1], block=BasicBlock)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 5)


def test_resnet_forward_other_input_sizes():
    model = resnet18()
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 10)


def test_config_overrides_nested():
    cfg = Config()
    _apply_overrides(cfg, {"train": {"epochs": 3}, "model": {"blocks": [1, 1, 1, 1]}})
    assert cfg.train.epochs == 3
    assert cfg.model.blocks == [1, 1, 1, 1]


def test_config_override_unknown_key_raises():
    with pytest.raises(KeyError):
        _apply_overrides(Config(), {"nope": 1})


def test_load_config_missing_file_gives_defaults(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.data.dataset == "cifar10"


def test_checkpoint_roundtrip(tmp_path):
    from resnet_mlops.evaluate import load_checkpoint

    model = resnet18(num_classes=10)
    path = tmp_path / "m.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"num_classes": 10, "blocks": [2, 2, 2, 2]},
            "classes": [str(i) for i in range(10)],
            "epoch": 0,
            "val_acc": 0.5,
        },
        path,
    )
    loaded, ckpt = load_checkpoint(path, "cpu")
    model.eval()  # match eval-mode running stats used by load_checkpoint
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(loaded(x), model(x))
    assert ckpt["classes"] == [str(i) for i in range(10)]
