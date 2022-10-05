import torch
from einops import rearrange
import numpy as np
from datasets.datasets import Rays_keys, Rays

def lift_gaussian(directions, t_mean, t_var, r_var, diagonal):

    mean = torch.unsqueeze(directions, dim = -2) * torch.unsqueeze(t_mean, dim = -1)
    d_norm_denominator = torch.sum(directions ** 2, dim = -1, keepdim = True) + 1e-10
    if diagonal:
        d_outer_diag = directions ** 2  # eq (16)
        null_outer_diag = 1 - d_outer_diag / d_norm_denominator
        t_cov_diag = torch.unsqueeze(t_var, dim=-1) * torch.unsqueeze(d_outer_diag,
                                                                      dim=-2)  # [B, N, 1] * [B, 1, 3] = [B, N, 3]
        xy_cov_diag = torch.unsqueeze(r_var, dim=-1) * torch.unsqueeze(null_outer_diag, dim=-2)
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = torch.unsqueeze(directions, dim=-1) * torch.unsqueeze(directions,
                                                                        dim=-2)  # [B, 3, 1] * [B, 1, 3] = [B, 3, 3]
        eye = torch.eye(directions.shape[-1], device=directions.device)  # [B, 3, 3]
        # [B, 3, 1] * ([B, 3] / [B, 1])[..., None, :] = [B, 3, 3]
        null_outer = eye - torch.unsqueeze(directions, dim=-1) * (directions / d_norm_denominator).unsqueeze(-2)
        t_cov = t_var.unsqueeze(-1).unsqueeze(-1) * d_outer.unsqueeze(-3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        xy_cov = t_var.unsqueeze(-1).unsqueeze(-1) * null_outer.unsqueeze(
            -3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        cov = t_cov + xy_cov
        return mean, cov




def cast_frustum_to_gaussian(directions, t0, t1, base_radius, diagonal, stable = True):
    ###按论文中的公式（7）
    if stable:
        mu = (t0+t1)/2
        hw = (t1-t0)/2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                            (3 * mu ** 2 + hw ** 2) ** 2)
        r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                    (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    else:
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2

    return lift_gaussian(directions, t_mean, t_var, r_var, diagonal)



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


def resample_along_rays(origins, directions, radii, t_samples, weights, randomized, ray_shape, stop_grad,
                        rasample_padding):

    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat([weights[...,:1], weights,weights[...,-1:]],dim = -1)
            weights_max = torch.maximum(weigthts_pad[...,:-1], weights_pad[...,1:])
            weights_blur = 0.5 * (weights_max[...,:-1] + weights_max[...,1:])

            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_samples,
                weights,
                t_samples.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[...,:1], weights, weights[...,-1:]], dim = -1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[...,1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_samples,
            weights,
            t_samples.shape[-1],
            randomized,
        )

    means, convs, = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, convs)

















