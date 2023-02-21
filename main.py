import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MNIST
from utils import get_device, set_seed, save_run, accuracy
from pcn import PCN

# Training Params
LR = 1e-4
BATCH_SIZE = 64
N_EPOCHS = 2
# Inference Params
INFERENCE_LR = 0.01
INFERENCE_ITERS_TRAIN = 10
INFERENCE_ITERS_TEST = 200


NETWORK = nn.Sequential(
    nn.Sequential(
        nn.Linear(10, 250),
        # nn.Tanh()
    ),
    nn.Sequential(
        nn.Linear(250, 250),
        # nn.Tanh()
    ),
    nn.Linear(250, 28*28)
)


def train(seed):
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
    optimizer = optim.Adam(model.network.parameters(), lr=LR)

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
                    f"Train loss: {model.loss:.5f}"
                    f"[{batch_id * len(img_batch)}/{len(train_loader.dataset)}]"
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
            test_acc += accuracy(label_preds, label_batch)

        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc / len(test_loader))
        print(f"\nAvg test accuracy: {test_accs[epoch-1]:.4f}\n")

    save_run(model, train_losses, test_losses, test_accs)
    return


if __name__ == "__main__":
    train(seed=0)
