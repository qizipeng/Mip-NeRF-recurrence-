import torch
from einops import rearrange
import numpy as np
from datasets.datasets import Rays_keys, Rays


def cast_rays(t_samples, origins, directions, radii, ray_shape, diagonal = True):

    t0 = t_samples[...,:-1]
    t1 = t_samples[...,1:]
    if ray_shape == "cone":
        gussian_fn = conical_frustum_to_gussian
    elif ray_shape == "cylinder":
        raise NotImplementedError
    else:
        assert False

    means, convs = gussian_fn(directions, t0, t1, radii, diagonal)
    means = means + torch.unsqueeze(origins, dim = -2)
    return means, convs






def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape):
    batch_size = origins.shape[0]

    t_samples = torch.linspace(0., 1., num_samples+1, device = origins.device)

    if disparity:
        t_samples = 1. / (1. / near * (1. - t_samples) + 1. / far * t_samples) ####这里是干什么？？？
    else:
        # t_samples = near * (1. - t_samples) + far * t_samples
        t_samples = near + (far - near) * t_samples

    if randomized:  ##？？？
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)
        lower = torch.cat([t_samples[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_samples = lower + (upper - lower) * t_rand

    else:
        # Broadcast t_samples to make the returned shape consistent.
        t_samples = torch.broadcast_to(t_samples, [batch_size, num_samples + 1])

    means, convs = cast_rays(t_samples, origins, directions, radii, ray_shape)

    return t_samples, (means,convs)