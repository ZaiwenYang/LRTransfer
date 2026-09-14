"""Parameter-wise metrics for an optimizer update."""

import math

import torch


@torch.no_grad()
def parameter_update_metrics(previous, current, learning_rate, optimizer_norm):
    """Return approximate ELR and the L2 distance between unit weight vectors.

    Norms cover the entire parameter tensor. Approximate ELR uses the weight
    norm before the update and the learning rate used by that optimizer step.
    A zero weight norm makes the corresponding metric undefined (NaN).
    """
    previous_norm = previous.norm().item()
    current_norm = current.norm().item()

    approximate_elr = (
        learning_rate * optimizer_norm / previous_norm
        if previous_norm > 0 else math.nan
    )
    normalized_weight_change = (
        (current / current_norm - previous / previous_norm).norm().item()
        if previous_norm > 0 and current_norm > 0 else math.nan
    )
    return approximate_elr, normalized_weight_change


def parameter_average(metrics):
    """Average parameter tensors equally, excluding wte and lm_head modules."""
    values = [
        value for name, value in metrics.items()
        if not {'wte', 'lm_head'}.intersection(name.split('.'))
    ]
    return sum(values) / len(values) if values else math.nan
