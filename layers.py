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
            device=None,
            dtype=None
    ) -> None:
        self.n_options = n_options
        self.real_out_features = out_features
        out_features = n_options * out_features
        self.errs = None
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        input (torch.Tensor): [batch_dim, in_features]
        z (torch.Tensor): [batch_dim, out_features]
        """
        preds = self.multiple_predictions(input)
        if z is None:
            average = torch.mean(preds, dim=-1)
            return average
        self.errs = self.get_errors(preds, z)
        probs = torch.softmax(self.errs, dim=1)
        return preds @ probs

    def multiple_predictions(self, input: torch.Tensor) -> torch.Tensor:
        pred = F.linear(input, self.weight, self.bias)  # [batch_size, out_features * n_probs]
        pred = pred.reshape(-1, self.real_out_features, self.n_options)  # [batch_size, out_features, n_probs]
        return pred

    def get_errors(self, pred: Tensor, z: Optional[Tensor]):
        z = z[:, :, None].reshape(-1, 1, self.real_out_features)
        pred = pred.reshape(-1, self.n_options, self.real_out_features)
        errs = torch.cdist(pred, z) # [batch_size, n_probs]
        return errs

    def error(self, input: torch.Tensor, z: torch.Tensor):
        """
        Parameters
        ----------
        input (torch.Tensor): [batch_dim, in_features]
        z (torch.Tensor): [batch_dim, out_features]

        Returns
        ---------
        (torch.Tensor): [batch_size, n_maps]
        """
        pred = self.multiple_predictions(input)
        errs = self.get_errors(pred, z)
        return errs

    TEMPERATURE = 1
    def free_energy_func(self, x, z):
        _, n_dims = x.shape
        errors = torch.cdist(x, self.multiple_predictions(z).reshape(-1, n_dims))
        return - torch.logsumexp(- self.TEMPERATURE * errors, dim=1)


class SequentialAttention(nn.Sequential):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input
