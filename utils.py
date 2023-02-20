import numpy as np
import torch
import random
from pathlib import Path
import os


def save_run(
        model, train_losses, test_losses, test_accs, path: str = "./model"
) -> None:
    directory = Path(path)
    train_loss_path = directory / "train_losses.npy"
    test_losses_path = directory / "test_losses.npy"
    test_accs_path = directory / "test_accs.npy"
    model_weights_path = directory / "weights.pth"
    os.makedirs(directory, exist_ok=True)
    np.save(train_loss_path, train_losses)
    np.save(test_losses_path, test_losses)
    np.save(test_accs_path, test_accs)
    model.save_weights(model_weights_path)


def get_device():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"USING DEVICE: {device_str}")
    return torch.device(device_str)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(pred_labels, true_labels):
    batch_size = pred_labels.size(0)
    correct = 0
    for b in range(batch_size):
        if torch.argmax(pred_labels[b, :]) == torch.argmax(true_labels[b, :]):
            correct += 1
    return correct / batch_size
