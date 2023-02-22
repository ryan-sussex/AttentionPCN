from typing import Union
import torch


class PCN(object):

    def __init__(
        self,
        network,
        dt: float = 0.01,
        device: Union[str, int] = "cpu"
    ):

        self.network = network.to(device)
        self.n_layers = len(self.network)
        self.n_nodes = self.n_layers + 1
        self.dt = dt
        self.n_params = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )
        self.device = device

    def reset(self):
        self.zero_grad()
        self.preds = [None] * self.n_nodes
        self.errs = [None] * self.n_nodes
        self.xs = [None] * self.n_nodes

    def reset_xs(self, prior, init_std):
        self.set_prior(prior)
        self.propagate_xs()
        for l in range(self.n_layers):
            # Doesn't this just overwrite the first two lines?
            self.xs[l] = torch.empty(self.xs[l].shape).normal_(
                mean=0, std=init_std).to(self.device)

    def set_obs(self, obs):
        self.xs[-1] = obs.clone()

    def set_prior(self, prior):
        self.xs[0] = prior.clone()

    def forward(self, x):
        return self.network(x)

    def propagate_xs(self):
        for l in range(1, self.n_layers):
            print(l)
            self.xs[l] = self.network[l - 1](self.xs[l - 1])

    def infer(
        self,
        obs: torch.Tensor,
        prior: torch.Tensor,
        n_iters: int,
        init_std: float = 0.05,
        test: bool = False
    ) -> torch.Tensor:
        """
        Runs n_iters of inference updates, calculating prediction errrors at
        each layer and using these to update the activities.

        After convergence (or early stopping) the weight gradients
        are calculated in order for the optimiser to perform gradient descent.
        """
        self.reset()
        self.set_prior(prior)
        self.propagate_xs()
        if test:
            self.reset_xs(prior, init_std)
        self.set_obs(obs)

        for t in range(n_iters):
            self.network.zero_grad()
            self.preds[-1] = self.network[self.n_layers -
                                          1](self.xs[self.n_layers - 1])
            # print(self.preds[-1].size())
            # print(self.xs[-1].size())
            self.errs[-1] = self.xs[-1] - self.preds[-1]

            for l in reversed(range(1, self.n_layers)):
                # Create predictions (activities)
                self.preds[l] = self.network[l-1](self.xs[l - 1], z=self.xs[l])
                err_func = (
                    lambda x: self.network[l-1][0].error(self.xs[l-1], x)
                )
                self.errs[l] = - torch.logsumexp(self.network[l-1][0].errs, dim=1)
                print("errors:", self.errs[l].size())
                print("preds:", self.preds[l].size())
                _, epsdfdx = torch.autograd.functional.vjp(
                    err_func, self.xs[l], self.errs[l])

                with torch.no_grad():
                    # Update x using gradient.
                    dx = epsdfdx
                    self.xs[l] = self.xs[l] + self.dt * dx

            if test:  # In test mode we need to update the first layer
                _, epsdfdx = torch.autograd.functional.vjp(
                    self.network[0], self.xs[0], self.errs[1])
                with torch.no_grad():
                    self.xs[0] = self.xs[0] + self.dt * epsdfdx

            if (t+1) != n_iters:
                self.clear_grads()

        self.set_weight_grads()
        return self.xs[0]

    def set_weight_grads(self):
        for l in range(self.n_layers):
            for w in self.network[l].parameters():
                dw = torch.autograd.grad(
                    self.preds[l + 1],
                    w,
                    - self.errs[l + 1],
                    allow_unused=True,
                    retain_graph=True
                )[0]
                w.grad = dw.clone()

    def zero_grad(self):
        self.network.zero_grad()

    def save_weights(self, path):
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path):
        self.network.load_state_dict(torch.load(path))

    def clear_grads(self):
        with torch.no_grad():
            for l in range(1, self.n_nodes):
                self.preds[l] = self.preds[l].clone()
                self.errs[l] = self.errs[l].clone()
                self.xs[l] = self.xs[l].clone()

    # @property
    # def free_energy(self) -> float:
    #     return (self.errs **2).mean().item()

    @property
    def loss(self) -> float:
        return (self.errs[-1]**2).mean().item()

    def __str__(self):
        return f"PCN(\n{self.network}\n"
