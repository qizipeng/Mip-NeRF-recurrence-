import torch
from pytorch.lightning import LightningModule
from models.mip_nerf import MipNerf
from models.mip import rearrange_render_image, distloss
from utils.metrics import calc_psnr
from datasets import dataset_dict

from utils.lr_schedule import MipLRDecay
from torch.utils.data import DataLoader
from utils.vis import stack_rgb, visualize_depth

class MipNeRFSystem(LightningModule):
    def __init__(self, hparams):
        self.save_hyperparameters(hparams)
        self.train_randomized = hparams["train.randomized"]
        self.val_randomized = hparams["val.randomized"]
        self.white_bkgd = hparams["train.white_bkgd"]
        self.val_chunk_size = hparams["val.chunk_size"]
        self.batch_size = self.hparams["train.batch_size"]

        self.mip_nerf = MipNerf(
            num_samples=hparams['nerf.num_samples'],
            num_levels=hparams['nerf.num_levels'],
            resample_padding=hparams['nerf.resample_padding'],
            stop_resample_grad=hparams['nerf.stop_resample_grad'],
            use_viewdirs=hparams['nerf.use_viewdirs'],
            disparity=hparams['nerf.disparity'],
            ray_shape=hparams['nerf.ray_shape'],
            min_deg_point=hparams['nerf.min_deg_point'],
            max_deg_point=hparams['nerf.max_deg_point'],
            deg_view=hparams['nerf.deg_view'],
            density_activation=hparams['nerf.density_activation'],
            density_noise=hparams['nerf.density_noise'],
            density_bias=hparams['nerf.density_bias'],
            rgb_activation=hparams['nerf.rgb_activation'],
            rgb_padding=hparams['nerf.rgb_padding'],
            disable_integration=hparams['nerf.disable_integration'],
            append_identity=hparams['nerf.append_identity'],
            mlp_net_depth=hparams['nerf.mlp.net_depth'],
            mlp_net_width=hparams['nerf.mlp.net_width'],
            mlp_net_depth_condition=hparams['nerf.mlp.net_depth_condition'],
            mlp_net_width_condition=hparams['nerf.mlp.net_width_condition'],
            mlp_skip_index=hparams['nerf.mlp.skip_index'],
            mlp_num_rgb_channels=hparams['nerf.mlp.num_rgb_channels'],
            mlp_num_density_channels=hparams['nerf.mlp.num_density_channels'],
            mlp_net_activation=hparams['nerf.mlp.net_activation']
        )

    def forward(self, batch_rays: torch.Tensor, randomized: bool, white_bkgd: bool):

        res = self.mip_nerf(batch_rays, randomized, white_bkgd)

        return res

    def setup(self, stage):
        dataset = dataset_dict(hparams['dataset_name'])

        self.train_dataset = dataset(
            data_dir = self.hparams["data_path"],
            split='train',
            white_bkgd=self.hparams['train.white_bkgd'],
            batch_type=self.hparams['train.batch_type'],
        )

        self.val_dataset = dataset(data_dir=self.hparams['data_path'],
                                   split='val',
                                   white_bkgd=self.hparams['val.white_bkgd'],
                                   batch_type=self.hparams['val.batch_type']
                                   )


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.mip_nerf.parameters(), lr = self.hparams['optimizer.lr_init'])

        scheduler = MipLRDecay(optimizer, self.hparams['optimizer.lr_init'], self.hparams['optimizer.lr_final'],
                               self.hparams['optimizer.max_steps'], self.hparams['optimizer.lr_delay_steps'],
                               self.hparams['optimizer.lr_delay_mult'])

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams['train.num_work'],
                          batch_size=self.hparams['train.batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        # must give 1 worker
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=1,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=True,
                          persistent_workers=True)


    def training_step(self, batch, batch_nb):

        rays, rgbs = batch
        ret = self(rays, self.train_randomized, self.white_bkgd)

        mask = rays.lossmult
        if self.hparams["loss.disable_multiscale_loss"]:
            mask = torch.ones_like(mask)
        losses =[]
        distlosses = []
        for (rgb, _, _,weights, t_samples) in ret:
            losses.append(
                (mask * (rgb - rgbs[..., :3]) ** 2).sum() / mask.sum())
            distlosses.append(distloss(weights, t_samples))

        mse_coarse, mse_fine = losses
        loss = self.hparams['loss.coarse_loss_mult'] * \
               (mse_coarse + 0.01 * distlosses[0]) + mse_fine + 0.01 * distlosses[-1]

        with torch.no_grad():
            psnrs =[]
            for(rgb,_,_,_,_) in ret:
                psnrs.append(calc_psnr(rbg, rgbs[...,:3]))
            psnr_coarse, psnr_fine = psnrs

        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_fine, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        _, rgbs = batch
        rgb_gt = rgbs[...,:3]
        coarse_rbg, fine_rgb, val_mask = self.render_image(batch)

        val_mse_coarse = (val_mask * (coarse_rgb - rgb_gt)
                          ** 2).sum() / val_mask.sum()
        val_mse_fine = (val_mask * (fine_rgb - rgb_gt)
                        ** 2).sum() / val_mask.sum()

        val_loss = self.hparams['loss.coarse_loss_mult'] * \
            val_mse_coarse + val_mse_fine

        val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)

        log = {'val/loss': val_loss, 'val/psnr': val_psnr_fine}
        stack = stack_rgb(rgb_gt, coarse_rgb, fine_rgb)  # (3, 3, H, W)
        self.logger.experiment.add_images('val/GT_coarse_fine',
                                          stack, self.global_step)

        return  log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


    def rendre_image(self, batch):
        rays, rgbs = batch
        _,height, width,_ = rgbs.shape
        single_image_rays, val_mask = rearrange_render_image(
            rays, self.val_chunk_size
        )
        coarse_rgb, fine_rgb = [], []
        distances = []
        with torch.no_grad():
            for batch_rays in single_image_rays:
                (c_rgb, _, _, _, _), (f_rgb, distance, _, _, _) = self(
                    batch_rays, self.val_randomized, self.white_bkgd)
                coarse_rgb.append(c_rgb)
                fine_rgb.append(f_rgb)
                distances.append(distance)

        coarse_rgb = torch.cat(coarse_rgb, dim=0)
        fine_rgb = torch.cat(fine_rgb, dim=0)
        distances = torch.cat(distances, dim=0)
        distances = distances.reshape(1, height, width)  # H W
        distances = visualize_depth(distances)
        self.logger.experiment.add_image('distance', distances, self.global_step)

        coarse_rgb = coarse_rgb.reshape(
            1, height, width, coarse_rgb.shape[-1])  # N H W C
        fine_rgb = fine_rgb.reshape(
            1, height, width, fine_rgb.shape[-1])  # N H W C

        return coarse_rgb, fine_rgb, val_mask










