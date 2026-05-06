import torch
import torch.distributed as dist
from itertools import repeat
from math import inf, sqrt
import numpy as np

# -----------------------------------------------------------------------------
# Muon optimizer

def optimal_quintic(l, u):
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        # Above this threshold, the equioscillating polynomials 
        # is numerically equal to...
        return (15/8)/u, (-10/8)/(u**3), (3/8)/(u**5)
    # This initialization becomes exact as l -> u
    q = (3*l + 1) / 4
    r = (l + 3) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3*b + np.array([-1, 1]) * 
                        sqrt(9*b**2 - 20*a*c)) / (10*c))
    return float(a), float(b), float(c)

target_slope = 0
def obj(l):
    a, b, c = optimal_quintic(l, 1)
    total = (a+b+c)
    a /= total; b /= total; c /= total
    local_argmin = np.sqrt((-3*b + sqrt(9*b**2 - 20*a*c)) / (10*c))
    local_min = a*local_argmin + b*local_argmin**3 + c*local_argmin**5
    return local_min / local_argmin - target_slope


def optimal_composition(l, num_iters, safety_factor_eps=0, cushion=0):
    u = 1
    assert 0 <= l <= u
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for iter in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion*u), u)
        if cushion*u > l:
            # Due to cushioning, this may be centered around 1 with 
            # respect to 0.024*u, u. Recenter it around 1 with respect 
            # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
            pl = a*l + b*l**3 + c*l**5
            pu = a*u + b*u**3 + c*u**5
            rescaler = 2/(pl + pu)
            a *= rescaler; b *= rescaler; c *= rescaler
        # Optionally incorporate safety factor here:
        if iter < num_iters - 1:  # don't apply to last polynomial
            a /= safety_factor; b /= safety_factor**3; c /= safety_factor**5
        coefficients.append((a, b, c))
        l = a*l + b*l**3 + c*l**5
        u = 2 - l
    return coefficients

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    coeffs_list = optimal_composition(l=1e-3, num_iters=steps, safety_factor_eps=1e-2, cushion=0.02)
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1): X = X.mT
    return X

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5, polar=PolarExpress)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5, compute_ns_err=False, layerwise_lr="Keller-Jordan",
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.compute_ns_err = compute_ns_err
        self.layerwise_lr = layerwise_lr
        self.rank = rank
        self.world_size = world_size
        self.total_error = 0.0
        # Track aggregated singular value statistics across parameters
        # (these will store the averaged per-matrix max/min singular values across all matrices)
        self.max_singular = 0.0
        self.min_singular = 0.0
        self.avg_singular = 0.0

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            # reset diagnostics for this step
            self.total_error = 0.0
            # for averaging per-matrix extremes we accumulate sums and counts locally
            local_max_sum = 0.0
            local_min_sum = 0.0
            local_avg_sum = 0.0
            local_count = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    if self.compute_ns_err:
                        # compute L2 (Frobenius) norm of (g @ g.T - I) in float32 for better numeric stability
                        err = torch.norm(
                            g.float() @ g.float().T - torch.eye(g.size(0), device=g.device, dtype=torch.float32)
                        ).item()
                        Gf = g.float()
                        r, c = Gf.size()
                        if r <= c:
                            err_mat = Gf @ Gf.T - torch.eye(r, device=Gf.device, dtype=torch.float32)
                        else:
                            err_mat = Gf.T @ Gf - torch.eye(c, device=Gf.device, dtype=torch.float32)
                        # print(err_mat.detach().cpu().numpy())
                        err = torch.norm(err_mat).item()
                        # print(f"Rank {self.rank}, param {i}, orthogonality error: {err:.6f}")
                        self.total_error += err
                        # compute singular values (max/min) of G for diagnostics
                        try:
                            s = torch.linalg.svdvals(g.float())
                            s_max = float(s.max().item())
                            s_min = float(s.min().item())
                            s_avg = float(s.mean().item())
                            local_max_sum += s_max
                            local_min_sum += s_min
                            local_avg_sum += s_avg
                            local_count += 1
                        except Exception:
                            # numerical issues shouldn't break training; ignore if SVD fails for this param
                            pass
                    if self.layerwise_lr == "Keller-Jordan":
                        g *= max(1, g.size(0)/g.size(1))**0.5
                    if self.layerwise_lr == "Moonlight":
                        g *= 0.2 * max(g.size(0), g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            if self.compute_ns_err:
                tot_err = torch.tensor(self.total_error, device='cuda', dtype=torch.float32)
                dist.all_reduce(tot_err, op=dist.ReduceOp.SUM)

                # aggregate per-matrix sums and counts across processes, then compute global averages
                sum_max_t = torch.tensor(local_max_sum, device='cuda', dtype=torch.float32)
                sum_min_t = torch.tensor(local_min_sum, device='cuda', dtype=torch.float32)
                sum_avg_t = torch.tensor(local_avg_sum, device='cuda', dtype=torch.float32)
                count_t = torch.tensor(local_count, device='cuda', dtype=torch.float32)
                # sum across ranks
                dist.all_reduce(sum_max_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_min_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_avg_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
                total_count = float(count_t.item())
                if total_count > 0:
                    self.max_singular = float((sum_max_t / count_t).item())
                    self.min_singular = float((sum_min_t / count_t).item())
                    self.avg_singular = float((sum_avg_t / count_t).item())
                    self.total_error = float((tot_err / count_t).item())
                else:
                    # fallback to zeros if nothing was measured
                    self.max_singular = 0.0
                    self.min_singular = 0.0
                    self.avg_singular = 0.0

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()
