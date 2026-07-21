import torch

from training import train_cnn


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.length = 2

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = torch.zeros(3, 512, 512)
        label = 0 if idx % 2 == 0 else 1
        return image, label


def test_role_training_uses_role_class_names_for_plotting(monkeypatch) -> None:
    captured = {}

    def fake_save_plots(history, model, val_loader, device, class_names, log_dir, model_name="default"):
        captured["class_names"] = class_names
        captured["model_name"] = model_name

    monkeypatch.setattr(train_cnn, "_save_cnn_plots", fake_save_plots)
    monkeypatch.setattr(train_cnn, "HumanoidDataset", FakeDataset)
    monkeypatch.setattr(train_cnn.torch, "save", lambda *args, **kwargs: None)

    train_cnn.train(
        data_dir="data",
        epochs=1,
        seed=42,
        save_path="tmp.pth",
        log_dir="tmp_logs",
        model_name="role",
    )

    assert captured["model_name"] == "role"
    assert captured["class_names"] == ["civilian", "medic", "engineer"]
