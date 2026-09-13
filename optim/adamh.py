import torch
from torch.optim import AdamW

class AdamH(AdamW):
    """
    AdamH Optimizer — backed by torch.optim.AdamW for speed.

    Update rule:
        w_{t+1} = R * Normalize(w_t - eta * R * Normalize(adam_update))

    where Normalize(x) = x / ||x||_2  (unit-norm, per parameter tensor)
    and   R is the radius hyperparameter (default: each param's initial L2 norm).

    Weights are constrained to lie on an L2-sphere of radius R, and the
    effective inner step is scaled by R as well.

    Strategy
    --------
    AdamW already computes:
        w_adamw = w_t - lr * adam_update   (plus its own decoupled weight-decay)

    so after AdamW.step() we can recover the raw Adam update direction as:
        adam_update_direction ∝ (w_t - w_adamw) / lr

    Then we overwrite the weight with:
        w_{t+1} = R * Normalize(w_t - η·R·Normalize(adam_update_direction))

    This lets us reuse AdamW's fused CUDA kernel for all moment bookkeeping
    while adding only two cheap normalisation ops per parameter.

    Args
    ----
    params       : iterable of parameters or param groups
    lr (eta)     : learning rate  (default: 1e-3)
    betas        : (beta1, beta2) Adam coefficients  (default: (0.9, 0.999))
    eps          : denominator stabiliser  (default: 1e-8)
    weight_decay : AdamW decoupled weight-decay coefficient  (default: 0.0)
                   Note: set to 0 (default) to match the original AdamH spec;
                   non-zero values let AdamW apply its own weight-decay before
                   the normalisation step.
    radius (R)   : sphere radius; weights are projected onto the R-sphere
                   and the inner step is scaled by R.
                   Default: -1.0 (sentinel) → each parameter's initial L2 norm.
    norm_dim     : dims for the L2 norm used in Normalize().
                   None  -> whole tensor treated as one vector  (default)
                   int / tuple -> norm computed over those dims only
                                  e.g. norm_dim=1 for per-row normalisation
    amsgrad      : use AMSGrad variant of AdamW  (default: False)
    fused        : use the fused CUDA kernel (requires CUDA, default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        radius: float = -1.0,       # sentinel: -1.0 → use each param's init norm
        norm_dim=None,
        amsgrad: bool = False,
        fused: bool = False,
    ):
        if radius != -1.0 and radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        self._adamh_radius   = radius
        self._adamh_norm_dim = norm_dim
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fused=fused,
        )

    @torch.no_grad()
    def _initialize_radii(self, group):
        radii = []
        for p in group["params"]:
            current_norm = p.detach().norm().clamp(min=1e-12)
            if group["radius"] < 0:
                # The default radius belongs to the parameter, not the group.
                radii.append(current_norm.item())
            else:
                radii.append(group["radius"])
                p.mul_(group["radius"] / current_norm)
        group["radii"] = radii

    def add_param_group(self, param_group):
        param_group.setdefault("radius",   getattr(self, "_adamh_radius",   -1.0))
        param_group.setdefault("norm_dim", getattr(self, "_adamh_norm_dim", None))
        super().add_param_group(param_group)
        self._initialize_radii(self.param_groups[-1])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if "radii" not in group:
                # Older checkpoints did not store per-parameter radii. The
                # loaded parameters already lie on their intended spheres.
                group["radii"] = [
                    p.detach().norm().clamp(min=1e-12).item()
                    for p in group["params"]
                ]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        snapshots = {
            p: p.detach().clone()
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        }

        super().step()

        for group in self.param_groups:
            lr = group["lr"]

            for p, radius in zip(group["params"], group["radii"]):
                if p not in snapshots:
                    continue

                w_t     = snapshots[p]
                w_adamw = p.detach()

                adam_update = w_t - w_adamw

                normed_update = adam_update / adam_update.norm().clamp(min=1e-12)
                p_new         = w_t - lr * radius * normed_update
                p.copy_(radius * p_new / p_new.norm().clamp(min=1e-12))

        return loss
