import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", 
                 vf_weight=0.1, adaptive_vf=True, vf_loss_type="combined_v3", distmat_margin=0.25, cos_margin=0.5):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
        self.vf_weight = vf_weight
        self.adaptive_vf = adaptive_vf
        self.vf_loss_type = vf_loss_type
        self.distmat_margin = distmat_margin
        self.cos_margin = cos_margin

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None, threshold=1e4):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, threshold).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, z_prime, features, optimizer_idx,
                global_step, last_layer=None, encoder_last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
                
            if z_prime is not None:
                if self.vf_loss_type == "combined_v3":
                    mcos_loss = nn.functional.cosine_similarity(z_prime, features, dim=-1)
                    mcos_loss = nn.functional.relu(1 - self.cos_margin - mcos_loss)
                    mcos_loss = torch.mean(mcos_loss)
                    
                    z_prime = nn.functional.normalize(z_prime, dim=-1)
                    features = nn.functional.normalize(features, dim=-1)
                    
                    cossim_z_prime = torch.matmul(z_prime, z_prime.transpose(1, 2))
                    cossim_features = torch.matmul(features, features.transpose(1, 2))
                    
                    mdms_loss = torch.abs(cossim_z_prime - cossim_features)
                    mdms_loss = nn.functional.relu(mdms_loss - self.distmat_margin)
                    mdms_loss = torch.mean(mdms_loss)
                    
                    vf_loss = mcos_loss + mdms_loss
                elif self.vf_loss_type == "mcos_only":
                    mcos_loss = 0.0
                    bsz = z_prime.shape[0]
                    for i, (z_prime_i, features_i) in enumerate(zip(z_prime, features)):
                        z_prime_i = nn.functional.normalize(z_prime_i, dim=-1)
                        features_i = nn.functional.normalize(features_i, dim=-1)
                        cossim = mean_flat(-(z_prime_i * features_i).sum(dim=-1))
                        mcos_loss += nn.functional.relu(1 - self.cos_margin - cossim)
                    mcos_loss /= bsz
                    
                    vf_loss = mcos_loss
                    
                    weighted_nll_loss = torch.tensor(0.0, device=inputs.device)
                    kl_loss = torch.tensor(0.0, device=inputs.device)
                    g_loss = torch.tensor(0.0, device=inputs.device)
                else:
                    raise ValueError(f"Unknown vf_loss_type: {self.vf_loss_type}")
                
                if self.adaptive_vf:
                    try:
                        vf_weight = self.calculate_adaptive_weight(nll_loss, vf_loss, last_layer=encoder_last_layer, threshold=1e8)
                    except RuntimeError:
                        assert not self.training
                        vf_weight = torch.tensor(1.0)
                    vf_loss = vf_weight * vf_loss
            else:
                vf_loss = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + self.vf_weight * vf_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/vf_loss".format(split): vf_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

