import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """Kendall, Gal & Cipolla (2018): L = sum_i exp(-s_i) L_i + s_i, s_i = log sigma_i^2 (learned)."""

    def __init__(self, tasks):
        super().__init__()
        self.tasks = list(tasks)
        self.log_vars = nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, losses):
        total = 0.0
        for i, t in enumerate(self.tasks):
            if t in losses:
                total = total + torch.exp(-self.log_vars[i]) * losses[t] + self.log_vars[i]
        return total

    def weights(self):
        return {t: float(torch.exp(-self.log_vars[i])) for i, t in enumerate(self.tasks)}


class FixedWeighting(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.w = dict(weights)

    def forward(self, losses):
        return sum(self.w.get(t, 1.0) * l for t, l in losses.items())

    def weights(self):
        return dict(self.w)
