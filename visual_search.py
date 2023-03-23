from pcn import PCN
from main import NETWORK, INFERENCE_ITERS_TEST, INFERENCE_LR, TEMPERATURE
from layers import AttentionLayer, GMMLayer, VisualSearch

from dataset import MNIST
from utils import get_device, set_seed, save_run, accuracy
from pcn import PCN
from layers import AttentionLayer, GMMLayer, VisualSearch

from dataset import MNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import torch

transform = T.ToPILImage()


dataset = MNIST("./data/")


N_OPTIONS = 5
NETWORK = nn.Sequential(
        GMMLayer(
            10, 28*28, n_options=1, temperature=TEMPERATURE, nonlinearity=torch.relu),
        AttentionLayer(
            28*28, 28*28, n_options=1, temperature=1, nonlinearity=torch.relu),
        VisualSearch(
            28*28, 28*28, n_options=N_OPTIONS, temperature=100, nonlinearity=None)
)
NETWORK = PCN(NETWORK, dt=0.1).load_weights("./model/weights.pth")



def generate_multiple_digits(dataset, n_options):
    start_indx = 0
    imgs = []
    labels = []
    for i in range(n_options):
        start_indx + i
        img, label = dataset[start_indx + i]
        imgs.append(img)
        labels.append(label)
    batch_img = torch.concat(imgs, axis=0)
    batch_img = batch_img[None, :]
    batch_label = labels[2][None, :]
    return batch_img, batch_label

def visual_search(
        pcn=NETWORK,
        dataset=dataset,
        n_options=N_OPTIONS
    ):
    batch_img, batch_label = generate_multiple_digits(dataset)
    preds = []
    for iters in range(1, 40):
        pcn.infer(
            obs=batch_img,
            prior=batch_label,
            n_iters=iters,
            test=False,
            test_pred=False
        )
        # Check what network is predicting "imagining"
        pred = pcn.network[-1].forward(
            pcn.xs[-2], 
            probabilities=pcn.attention[3][0]
        )
        preds.append(pred.reshape(n_options * 28, 28))
    return preds