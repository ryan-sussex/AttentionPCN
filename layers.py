from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AttentionLayer(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            n_options: int = 2,
            temperature: int = 1,
            device=None,
            dtype=None
    ) -> None:
        self.n_options = n_options
        self.real_out_features = out_features
        out_features = n_options * out_features
        self.temperature = temperature
        super().__init__(in_features, out_features, bias, device, dtype)

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
        if probabilities is None:
            average = torch.mean(preds, dim=-1)
            return average
        return preds @ probabilities

    def multiple_predictions(self, input: Tensor) -> Tensor:
        pred = F.linear(input, self.weight, self.bias)
        # [batch_size, out_features * n_probs]
        pred = pred.reshape(-1, self.real_out_features, self.n_options)
        # [batch_size, out_features, n_probs]
        return pred

    def free_energy_func(self, x: Tensor, z: Tensor) -> Tensor:
        # Make multiple predictions
        predictions = self.multiple_predictions(z)
        # Fiddling with shapes
        n_batch, n_dims = x.shape
        x = x[:, None, :]
        predictions = predictions.reshape(n_batch, self.n_options, n_dims)
        # Calculate the error for each different prediction
        self.errs = torch.cdist(x, predictions)
        # Calculate free energy based on log sum exp of errrors
        inv_T = 1 / self.temperature
        return - inv_T * torch.logsumexp(- self.temperature * self.errs, dim=2)


class GMMLayer(AttentionLayer):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            n_options: int = 2,
            temperature: int = 1,
            device=None,
            dtype=None
        ) -> None:
        super().__init__(in_features, out_features, bias, n_options, temperature, device, dtype)
        nn.Linear.__init__(self, in_features, self.real_out_features)
        self.n_options = in_features

    def multiple_predictions(self, input: Tensor) -> Tensor:
        mask = torch.eye(input.size(1))
        preds = [
            F.linear(input * mask[i, :], self.weight, self.bias)
            for i in range(mask.shape[1])
        ]
        preds = torch.stack(preds)
        return preds.reshape(-1, self.real_out_features, self.n_options)