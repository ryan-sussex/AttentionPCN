from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        n_options: int = 2,
        temperature: Optional[int] = 1,
        nonlinearity = F.tanh,
        device=None,
        dtype=None,
    ) -> None:
        nn.Module.__init__(self)
    
        self.n_options = n_options
        for i in range(n_options):
            self.__setattr__(f"linear_{i}", nn.Linear(in_features, out_features, bias=bias, device=device))
        # self.real_out_features = out_features
        # out_features = n_options * out_features
        self.temperature = temperature
        self.attention = None
        self.nonlinearity = nonlinearity

    def forward(
            self,
            input: Tensor,
            probabilities: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        input (Tensor): [batch_dim, in_features]
        probabilities (Tensor): [batch_dim, n_options]

        Returns
        --------
        Tensor: [batch_dim, out_features]
        """
        preds = self.multiple_predictions(input)
        preds = self.apply_nonlinearity(preds)
        if probabilities is None:
            average = torch.mean(preds, dim=-1)
            return average
        out = torch.einsum('bpi,bi->bp', preds, probabilities)
        out = out.reshape(input.shape[0], -1)
        return out
    
    def apply_nonlinearity(self, input: torch.Tensor):
        if self.nonlinearity is None:
            return input
        return self.nonlinearity(input)

    def multiple_predictions(self, input: Tensor) -> Tensor:
        preds = [
            self.__getattr__(f"linear_{i}")(input)
            for i in range(self.n_options)
        ]
        preds = torch.stack(preds, dim=2)
        return preds

    def free_energy_func(
            self,
            x: Tensor,
            z: Tensor,
            record_att_weights: bool = True
        ) -> Tensor:
        # Make multiple predictions
        predictions = self.multiple_predictions(z)
        predictions = self.apply_nonlinearity(predictions)
        # Fiddling with shapes
        x = x[:, None, :]  # (n_batch, 1, dim)
        predictions = predictions.transpose(1, 2)  # (n_batch, n_preds, dim)
        # Calculate the error for each different prediction
        self.errs = torch.cdist(x, predictions)  # (n_batch, n_preds)
        # Optionally store what the network is paying attention to
        if record_att_weights:
            self.attention = torch.softmax(-self.temperature * self.errs, dim=2)
        # Calculate free energy based on log sum exp of errrors
        inv_T = 1 / self.temperature
        return -inv_T * torch.logsumexp(-self.temperature * self.errs, dim=2)


class GMMLayer(AttentionLayer, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        n_options: int = 2,
        temperature: int = 1,
        nonlinearity = F.tanh,
        device=None,
        dtype=None,
    ) -> None:
        AttentionLayer.__init__(
            self,
            in_features,
            out_features,
            bias,
            n_options,
            temperature,
            nonlinearity=nonlinearity,
            device=device,
            dtype=dtype,
        )
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        self.n_options = in_features
        mask = torch.eye(self.n_options, device=device)
        self.register_buffer("mask", mask, persistent=False)

    def multiple_predictions(self, input: Tensor) -> Tensor:
        preds = [
            F.linear(input * self.mask[i, :], self.weight, None)
            for i in range(self.mask.shape[1])
        ]
        preds = torch.stack(preds, dim=2)
        return preds


class VisualSearch(AttentionLayer, nn.Linear):
    def __init__(
            self, 
            in_features: int, 
            out_features: int,
            bias: bool = False, 
            n_options: int = 2, 
            temperature: Optional[int] = 1, 
            nonlinearity=F.tanh, 
            device=None, 
            dtype=None
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            n_options,
            temperature,
            nonlinearity,
            device,
            dtype
        )
        nn.Linear.__init__(self, in_features, out_features, bias=False)
    
    def _pad_prediction(self, pred: Tensor, pos: int) -> Tensor:
        batch_size, out_features = pred.shape
        padded = torch.zeros(
            (batch_size, out_features * self.n_options), 
            device=pred.device
        )
        padded[:, pos * out_features: (pos + 1) * out_features] = pred
        return padded

    def multiple_predictions(self, input: Tensor) -> Tensor:
        preds = [
            self._pad_prediction(
                F.linear(input, self.weight, None), pos=i
            )
            for i in range(self.n_options)
        ]
        preds = torch.stack(preds, dim=2)
        return preds


def pad(tensor: Tensor, n_copies: int, pos: int) -> Tensor:
    batch_size, out_features = tensor.shape
    padded = torch.zeros(
        (batch_size, out_features * n_copies), 
        device=tensor.device
    )
    padded[:, pos * out_features: (pos + 1) * out_features] = tensor
    return padded
