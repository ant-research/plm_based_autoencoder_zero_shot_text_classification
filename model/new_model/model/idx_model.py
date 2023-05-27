import torch.nn as nn
from model.new_model.utils.loss import CEFocalLoss
from model.new_model.model.vq_vae_idx import VAE


class ConFixVQVAE_Idx(nn.Module):
    def __init__(self, config, device, label_matrix, gpu_tracker, emb_layer, bos=None, **kwargs):
        super().__init__()
        self.config = config
        self.gpu_tracker = gpu_tracker  # gpu memory tracker
        self.label_matrix = label_matrix  # label embedding matrix [label_count, emb_size]
        self.discri_type = config.discri_type  # discriminator type
        self.vae_type = config.vae_type
        self.vq_vae = VAE[self.vae_type](config, device, gpu_tracker=self.gpu_tracker, bos=bos,
                                         label_mat=self.label_matrix,
                                         emb_layer=emb_layer)
        #self.classifier_loss = nn.CrossEntropyLoss()
        self.classifier_loss = CEFocalLoss()

    def _init_model(self):
        pass

    def forward(self, x, x_pad_mask=None, easy=True):
        '''
        x: [batch_size, seq_len, emb_size]
        x: [batch_size, seq_len]
        '''
        result, z_t, z_p, loss_p, prob = self.vq_vae(x, x_pad_mask, easy=easy)
        return prob, result, z_t, z_p, loss_p

    def loss_function(self, x, recon_x, loss_p, output, y=None, x_pad_mask=None, epoch=1, generate=False, unlabel=False):
        '''
        x: origin input
        recon_x: reconstruction x
        loss_p: dispersing vq loss
        output: prob output
        y: origin y
        x_pad_mask: origin input mask
        generate: whether the x is generated or from the dataset
        unlabel: whether the x has not y
        '''
        discriminator_loss = None
        vq_loss = self.vq_vae.loss_function(x, recon_x, loss_p, x_pad_mask=x_pad_mask, generate=generate)
        if unlabel is False:
            discriminator_loss = self.classifier_loss(output, y)
            print('discriminator loss', discriminator_loss)
            loss = discriminator_loss + vq_loss * self.config.other_coef
        else:
            loss = vq_loss
        return loss, discriminator_loss

    def sample_result(self, index, device, n=10):
        result = self.vq_vae.sample_result(index, device, n=n)
        return result
