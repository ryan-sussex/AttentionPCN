import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MNIST
from utils import get_device, set_seed, save_run, accuracy
from pcn import PCN
from layers import AttentionLayer, GMMLayer, VisualSearch

# Training Params
LR = 1e-4
BATCH_SIZE = 128
N_EPOCHS = 4
# Inference Params
TEMPERATURE = 10
INFERENCE_LR = 0.1
INFERENCE_ITERS_TRAIN = 20
INFERENCE_ITERS_TEST = 200


# Note attention layer with 1 option is a standard layer. 
NETWORK = nn.Sequential(
        GMMLayer(
            10, 250, n_options=1, temperature=TEMPERATURE),
        AttentionLayer(
            250, 28*28, n_options=1, temperature=1),
        VisualSearch(
            28*28, 28*28, n_options=1, temperature=1, nonlinearity=None)
)


def train(seed, weights_path=None):
    set_seed(seed)
    device = get_device()

    train_data = MNIST(train=True)
    test_data = MNIST(train=False)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = PCN(
        network=NETWORK,
        device=device,
        dt=INFERENCE_LR
    )
    if weights_path:
        model.load_weights(weights_path)

    optimizer = optim.Adam(model.network.parameters(), lr=LR, weight_decay=1)

    train_losses, test_losses = [], []
    test_accs = []
    log_every = 100

    for epoch in range(1, N_EPOCHS+1):
        print(f"Epoch {epoch}\n-------------------------------")
        train_loss = 0
        for batch_id, (img_batch, label_batch) in enumerate(train_loader):
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            model.infer(
                obs=img_batch,
                prior=label_batch,
                n_iters=INFERENCE_ITERS_TRAIN,
                test=False
            )
            optimizer.step()
            train_loss += model.loss

            if batch_id % log_every == 0:
                print(
                    f"Train reconstruction loss: {model.loss:.5f}"
                    f"[{batch_id * len(img_batch)}/"
                    f"{len(train_loader.dataset)}]"
                )
                print(
                    f"Train label loss: {model.average_free_energy(1):.5f}"
                    f"[{batch_id * len(img_batch)}/"
                    f"{len(train_loader.dataset)}]"
                )
                print(
                    f"Total free energy: {model.total_free_energy:.5f}"
                    f"[{batch_id * len(img_batch)}/"
                    f"{len(train_loader.dataset)}]"
                )

        test_loss, test_acc = (0, 0)
        for batch_id, (img_batch, label_batch) in enumerate(test_loader):
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            label_preds = model.infer(
                obs=img_batch,
                prior=label_batch,
                n_iters=INFERENCE_ITERS_TEST,
                test=True
            )
            test_loss += model.loss
            if batch_id % log_every == 0:
                print(
                    f"Test reconstruction loss: {model.loss:.5f}"
                    f"[{batch_id * len(img_batch)}/"
                    f"{len(train_loader.dataset)}]"
                )
                print(
                    f"Test label loss: {model.average_free_energy(1):.5f}"
                    f"[{batch_id * len(img_batch)}/"
                    f"{len(train_loader.dataset)}]"
                )

            test_acc += accuracy(label_preds, label_batch)

        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc / len(test_loader))
        print(f"\nAvg test accuracy: {test_accs[epoch-1]:.4f}\n")

    save_run(model, train_losses, test_losses, test_accs)
    return


if __name__ == "__main__":
    WEIGHTS_PATH = "./model/weights.pth"
    # WEIGHTS_PATH=None
    train(seed=0, weights_path=WEIGHTS_PATH)
