import torch
import torch.nn as nn
from typing import Union, Tuple
import numpy as np
from model.new_model.utils.loss import CEFocalLoss
from model.new_model.modules.vq_vae import VAE
from model.new_model.modules.semi_vq_vae import Semi_VQ_VAE
from model.new_model.model.vae_model import VQVAEBase, FixVQVAE
from model.new_model.modules.classifier import Classifier
from model.new_model.modules.gcn import GraphModel


class Model(nn.Module):
    """
    double VQ VAE model, 适用于decoder输出概率以及输出vector情况
    """
    def __init__(self, config, device, label_matrix, gpu_tracker, emb_layer, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        self.label_matrix = label_matrix
        self.gpu_tracker = gpu_tracker
        self.emb_layer = emb_layer

        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        '''
        build models
        '''
        self.vq_vae = VAE[self.config.vae_type](self.config,
                                                self.device,
                                                gpu_tracker=self.gpu_tracker,
                                                emb_layer=self.emb_layer,
                                                label_mat=self.label_matrix,
                                                **kwargs)
        
        # whether have classifier
        if 'Fix' not in self.config.vae_type:
            self.is_classifier = True
            self.classifier = Classifier[self.config.classifier_type](self.config, self.label_matrix)
        else:
            self.is_classifier = False
            # self.classifier = None
            self.classifier_loss_func = CEFocalLoss()

    def forward(self, input_idx, x_pad_mask=None, freeze_encoder=False, **kwargs):
        '''
        x: torch.Tensor [B, S]
        x_pad_mask: mask [B, S]
        freeze_encoder： 用于预训练之后的分类器的训练
        '''
        result = self.vq_vae(input_idx, x_pad_mask, **kwargs)
        # 如果有classifier，将z_t输进去做分类
        if self.is_classifier:
            z_t = result['z_t']
            if freeze_encoder:
                z_t = z_t.detach()
            else:
                pass
            prob = self.classifier(z_t)
            result['prob'] = prob
        return result

    def loss_function(self, model_output_dict, x, y, x_pad_mask=None, dis_loss=True, **kwargs):
        loss_dict = {}
        vq_loss = self.vq_vae.loss_function(model_output_dict,
                                            x=x,
                                            x_pad_mask=x_pad_mask,
                                            **kwargs)
        loss_dict['vq_loss'] = vq_loss
        
        if dis_loss:
            prob = model_output_dict['prob']
            dis_loss = self.get_dis_loss(prob, y)
            loss_dict['dis_loss'] = dis_loss
        return loss_dict

    def get_dis_loss(self, prob, y):
        if self.is_classifier:
            dis_loss = self.classifier.loss_function(prob, y)
        else:
            dis_loss = self.classifier_loss_func(prob, y)
        return dis_loss

    def pretrain_decoder(self, input_idx, x_pad_mask=None, **kwargs):
        result_dict = {}
        output_prob = self.vq_vae.decoder.pretrain(input_idx, x_pad_mask)
        result_dict['recon_result'] = output_prob
        return result_dict


class Semi_Model(nn.Module):
    """
    Semi supervised model, move distilling vq vae, this model need a discriminator to train
    loss:
        reconstruction loss
        dispserse loss
        classifier loss
    """
    def __init__(self, config, device, label_matrix, gpu_tracker, emb_layer, adj_parent, adj_child, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        self.label_matrix = label_matrix
        self.gpu_tracker = gpu_tracker
        self.emb_layer = emb_layer
        self.adj_parent = adj_parent
        self.adj_child = adj_child
        self.disper_num = self.config.disper_num

        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        '''
        build models
        '''
        # generator
        self.vq_vae = Semi_VQ_VAE[self.config.vae_type](self.config,
                                                        self.device,
                                                        gpu_tracker=self.gpu_tracker,
                                                        emb_layer=self.emb_layer,
                                                        label_mat=self.label_matrix,
                                                        **kwargs)
        # classifier: input z_t and label matrix, output probability
        self.classifier = Classifier[self.config.classifier_type](self.config, adj_parent=self.adj_parent)
        # garph model: input label matrix and output new label feature
        self.graph_model = GraphModel[self.config.graph_type](self.adj_parent, self.adj_child, self.label_matrix,
                                                              **kwargs)

    def forward(self, input_idx, x_pad_mask=None, **kwargs):
        '''
        x: torch.Tensor [B, S]
        x_pad_mask: mask [B, S]
        freeze_encoder： 用于预训练之后的分类器的训练
        '''
        result = self.vq_vae(input_idx, x_pad_mask, **kwargs)
        z_t = result['z_t']

        # get label matrix
        label_matrix = self.graph_model(train=True)

        # 分类结果
        prob = self.classifier(z_t, label_matrix)
        result['prob'] = prob
        return result

    def loss_function(self, model_output_dict, x, y, x_pad_mask=None, dis_loss=True, **kwargs):
        loss_dict = {}
        vq_loss = self.vq_vae.loss_function(model_output_dict,
                                            x=x,
                                            x_pad_mask=x_pad_mask,
                                            **kwargs)
        loss_dict['vq_loss'] = vq_loss
        
        if dis_loss:
            prob = model_output_dict['prob']
            class_loss = self.get_class_loss(prob, y)
            loss_dict['classifier_loss'] = class_loss
        return loss_dict

    def get_class_loss(self, prob, y):
        class_loss = self.classifier.loss_function(prob, y)
        return self.config.classifier_coef * class_loss

    def get_label_matrix(self):
        """
        return label matrix after GAT
        """
        return self.graph_model(train=False)

    def sample_labels(self, n: int, y_idx: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample some labels from graph model to train the discriminator
        """
        label_matrix = self.get_label_matrix()
        if y_idx is None:
            y_idx = np.random.randint(0, self.disper_num, size=n)  # type: ignore
            y_idx = torch.from_numpy(y_idx).to(label_matrix.device)
        sample_label = label_matrix[y_idx, :]
        y = torch.ones(n)
        return sample_label, y

    def sample_result(self, y_idx, device, n=1):
        """Returns embedding tensor for a batch of indices."""
        codebook_emb_t, generate_label = self.sample_labels(y_idx=y_idx, n=0)  # [batch_size, distill_size]
        # random choose n samples from emb p
        n = y_idx.shape[0]

        if self.vq_vae.p_vq_type == 'DVQ':
            # 如果是DVQ, 
            p_idx_list = []
            emb_p_list = []
            for emb_p in self.vq_vae.emb_p.vq_layers:
                p_idx = np.random.randint(0, self.disper_num, size=n)
                p_idx = torch.from_numpy(p_idx).to(device)  # [batch_size, disperse_size]
                codebook_emb_p = emb_p.quantize_embedding(p_idx)  # [batch_size, disperse_size]
                p_idx_list.append(p_idx.unsqueeze(1))
                emb_p_list.append(codebook_emb_p.unsqueeze(1))
            codebook_emb_p = torch.cat(emb_p_list, dim=1)  # [batch_size, decomp_num, disperse_size]
            p_idx = torch.cat(p_idx_list, dim=1)  # [batch_size, decomp_num]
        else:
            p_idx = np.random.randint(0, self.disper_num, size=n)
            p_idx = torch.from_numpy(p_idx).to(device)
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx).unsqueeze(1)  # [batch_size, disperse_size]
        # result
        z = torch.cat([codebook_emb_t.unsqueeze(1), codebook_emb_p], dim=1).to(device)
        result = self.vq_vae.decoder(None, z, is_training=False)  # [batch_size, emb_size]
        return result.detach(), y_idx, p_idx  # 截断梯度流

    def sample_all(self, y_idx, device):
        """
        Returns embedding tensor for a batch of indices.
        y_idx [1]
        """
        if self.vq_vae.p_vq_type == 'DVQ':
            result, y_idx, p_idx = self.sample_result(y_idx=y_idx, device=device)
        else:
            codebook_emb_t, generate_label = self.sample_labels(y_idx=y_idx, n=0)  # [batch_size, distill_size]
            p_idx = torch.arange(0, self.disper_num, 1).to(device)
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx)  # [disper num, disperse_size]
            # result
            result = torch.cat([codebook_emb_t.unsqueeze(1).repeat(codebook_emb_p.shape[0], 1, 1),
                                codebook_emb_p.unsqueeze(1)], dim=1).to(device)
            result = self.vq_vae.decoder(None, result, is_training=False)  # [batch_size, emb_size]
        return result.detach(), y_idx, p_idx  # 截断梯度流


class New_Model(nn.Module):
    """
    Semi supervised model, move distilling vq vae, this model need a discriminator to train
    loss:
        reconstruction loss
        dispserse loss
        classifier loss
    """
    def __init__(self, config, device, label_matrix, gpu_tracker, emb_layer, emb_layer_p, adj_parent, adj_child, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        self.label_matrix = label_matrix
        self.gpu_tracker = gpu_tracker
        self.emb_layer = emb_layer
        self.emb_layer_p = emb_layer_p
        self.adj_parent = adj_parent
        self.adj_child = adj_child
        self.disper_num = self.config.disper_num

        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        '''
        build models
        '''
        # generator

        self.vq_vae = VQVAEBase(self.config,
                                self.device,
                                gpu_tracker=self.gpu_tracker,
                                emb_layer=self.emb_layer,
                                emb_layer_p=self.emb_layer_p,
                                label_mat=self.label_matrix,
                                **kwargs)
        # classifier: input z_t and label matrix, output probability
        self.classifier = Classifier[self.config.classifier_type](self.config, adj_parent=self.adj_parent)
        # garph model: input label matrix and output new label feature
        self.graph_model = GraphModel[self.config.graph_type](self.adj_parent, self.adj_child, self.label_matrix,
                                                              encoder_output_size=self.config.encoder_output_size,
                                                              **kwargs)

    def forward(self, input_idx, x_pad_mask=None, y=None, **kwargs):
        '''
        x: torch.Tensor [B, S]
        x_pad_mask: mask [B, S]
        freeze_encoder： 用于预训练之后的分类器的训练
        '''
        with torch.cuda.amp.autocast():
            # get label matrix
            label_matrix = self.graph_model(train=True)
            
            if y is None:
                result_dict = self.vq_vae(input_idx, x_pad_mask, y=y, **kwargs)
            else:
                result_dict = self.vq_vae(input_idx, x_pad_mask, y=y, label_matrix=label_matrix, **kwargs)
            z_t = result_dict['enc_outdict_t']['pooled_memory']

            # 分类结果
            classfier_result = self.classifier(z_t, label_matrix, y=y)
            result_dict['classifier'] = classfier_result
            return result_dict

    def get_label_matrix(self) -> torch.Tensor:
        """
        return label matrix after GAT
        """
        return self.graph_model(train=False)

    def sample_labels(self, n: int, y_idx: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample some labels from graph model to train the discriminator
        """
        label_matrix = self.get_label_matrix().detach()
        # if no y idx, randomly sampled some y
        if y_idx is None:
            sampled_y_idx = np.random.randint(0, self.disper_num, size=n)
            y_idx = torch.from_numpy(sampled_y_idx).to(label_matrix.device)
        sample_label = label_matrix[y_idx, :].contiguous()
        y = torch.ones(n)
        return sample_label, y

    def sample_result(self, y_idx, device, n=1):
        """Returns embedding tensor for a batch of indices."""
        codebook_emb_t, generate_label = self.sample_labels(y_idx=y_idx, n=0)  # [batch_size, distill_size]
        # random choose n samples from emb p
        n = y_idx.shape[0]

        if self.vq_vae.p_vq_type in ['GS', 'DVQ']:
            # 如果是DVQ, 
            p_idx_list = []
            for i in range(self.config.decompose_number):
                p_idx = np.random.randint(0, self.disper_num, size=n)
                p_idx = torch.from_numpy(p_idx).to(device)  # [batch_size, disperse_size]
                p_idx_list.append(p_idx.unsqueeze(1))
            p_idx = torch.cat(p_idx_list, dim=1)  # [batch_size, decomp_num]
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx)
        else:
            p_idx = np.random.randint(0, self.disper_num, size=n)
            p_idx = torch.from_numpy(p_idx).to(device)
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx).unsqueeze(1)  # [batch_size, disperse_size]
        # result
        # print('emb t shape', codebook_emb_t.shape, codebook_emb_p.shape, 'n', n, y_idx)
        z = torch.cat([codebook_emb_t.unsqueeze(1), codebook_emb_p], dim=1).to(device)
        result_dict = self.vq_vae.decoder(inputs=None, memory=z, generate=True)  # [batch_size, emb_size]
        return result_dict, y_idx, p_idx  # 截断梯度流

    def sample_all(self, y_idx, device, one_by_one=False):
        """
        Returns embedding tensor for a batch of indices.
        y_idx [1]
        """
        if self.vq_vae.p_vq_type in ['GS', 'DVQ'] and self.config.decompose_number > 1:
            result_dict, y_idx, p_idx = self.sample_result(y_idx=y_idx, device=device)
        else:
            codebook_emb_t, generate_label = self.sample_labels(y_idx=y_idx, n=0)  # [batch_size, distill_size]
            p_idx = torch.arange(0, self.disper_num, 1).to(device).unsqueeze(1)
            # print(p_idx)
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx)  # [disper num, disperse_size]
            # result
            result = torch.cat([codebook_emb_t.unsqueeze(1).repeat(codebook_emb_p.shape[0], 1, 1),
                                codebook_emb_p], dim=1).to(device)
            if one_by_one is True:
                result_dict = {}
                for i in range(result.shape[0]):
                    one_result_dict = self.vq_vae.decoder(inputs=None, memory=result[i, :, :].unsqueeze(0), generate=True)  # [batch_size, emb_size]
                    if result_dict == {}:
                        result_dict['pred_idx'] = one_result_dict['pred_idx']
                    else:
                        result_dict['pred_idx'] = torch.cat([result_dict['pred_idx'], one_result_dict['pred_idx']], dim=0)
            else:
                result_dict = self.vq_vae.decoder(inputs=None, memory=result, generate=True)  # [batch_size, emb_size]
        return result_dict, y_idx, p_idx  # 截断梯度流


class Fix_Model(nn.Module):
    """
    Semi supervised model, move distilling vq vae, this model need a discriminator to train
    loss:
        reconstruction loss
        dispserse loss
        classifier loss
    """
    def __init__(self, config, device, label_matrix, gpu_tracker, emb_layer, emb_layer_p, adj_parent, adj_child, **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        self.label_matrix = label_matrix
        self.gpu_tracker = gpu_tracker
        self.emb_layer = emb_layer
        self.emb_layer_p = emb_layer_p
        self.adj_parent = adj_parent
        self.adj_child = adj_child
        self.disper_num = self.config.disper_num

        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        '''
        build models
        '''
        # generator

        self.vq_vae = FixVQVAE(self.config,
                                self.device,
                                gpu_tracker=self.gpu_tracker,
                                emb_layer=self.emb_layer,
                                emb_layer_p=self.emb_layer_p,
                                label_mat=self.label_matrix,
                                **kwargs)
        # garph model: input label matrix and output new label feature
        self.graph_model = GraphModel[self.config.graph_type](self.adj_parent, self.adj_child, self.label_matrix,
                                                              encoder_output_size=self.config.encoder_output_size,
                                                              **kwargs)

    def forward(self, input_idx, x_pad_mask=None, y=None, **kwargs):
        '''
        x: torch.Tensor [B, S]
        x_pad_mask: mask [B, S]
        freeze_encoder： 用于预训练之后的分类器的训练
        '''
        with torch.cuda.amp.autocast():
            # get label matrix
            label_matrix = self.graph_model(train=True)
            
            result_dict = self.vq_vae(input_idx, x_pad_mask, y=y, label_matrix=label_matrix, **kwargs)
            
            # 分类结果
            result_dict['classifier'] = {
                'loss': result_dict['quantizer_out_t']['classifier_loss'],
                'prob': -result_dict['quantizer_out_t']['distances']
            }
            return result_dict


    def get_label_matrix(self) -> torch.Tensor:
        """
        return label matrix after GAT
        """
        return self.graph_model(train=False)

    def sample_labels(self, n: int, y_idx: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample some labels from graph model to train the discriminator
        """
        label_matrix = self.get_label_matrix().detach()
        # if no y idx, randomly sampled some y
        if y_idx is None:
            sampled_y_idx = np.random.randint(0, self.disper_num, size=n)
            y_idx = torch.from_numpy(sampled_y_idx).to(label_matrix.device)
        sample_label = label_matrix[y_idx, :].contiguous()
        y = torch.ones(n)
        return sample_label, y
    
    def sample_disperse(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample some labels from graph model to train the discriminator
        """
        # if no y idx, randomly sampled some y
        if self.vq_vae.p_vq_type in ['GS', 'DVQ']:
            # 如果是DVQ, 
            p_idx_list = []
            for i in range(self.config.decompose_number):
                p_idx = np.random.randint(0, self.disper_num, size=n)
                p_idx = torch.from_numpy(p_idx).to(device)  # [batch_size, disperse_size]
                p_idx_list.append(p_idx.unsqueeze(1))
            p_idx = torch.cat(p_idx_list, dim=1)  # [batch_size, decomp_num]
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx)
        else:
            p_idx = np.random.randint(0, self.disper_num, size=n)
            p_idx = torch.from_numpy(p_idx).to(device)
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx).unsqueeze(1)  # [batch_size, disperse_size]
        return codebook_emb_p

    def sample_result(self, y_idx, device, n=1, label_text=None):
        """Returns embedding tensor for a batch of indices."""
        codebook_emb_t, generate_label = self.sample_labels(y_idx=y_idx, n=0)  # [batch_size, distill_size]
        # random choose n samples from emb p
        n = y_idx.shape[0]

        if self.vq_vae.p_vq_type in ['GS', 'DVQ']:
            # 如果是DVQ, 
            p_idx_list = []
            for i in range(self.config.decompose_number):
                p_idx = np.random.randint(0, self.disper_num, size=n)
                p_idx = torch.from_numpy(p_idx).to(device)  # [batch_size, disperse_size]
                p_idx_list.append(p_idx.unsqueeze(1))
            p_idx = torch.cat(p_idx_list, dim=1)  # [batch_size, decomp_num]
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx)
        else:
            p_idx = np.random.randint(0, self.disper_num, size=n)
            p_idx = torch.from_numpy(p_idx).to(device)
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx).unsqueeze(1)  # [batch_size, disperse_size]
        # result
        # print('emb t shape', codebook_emb_t.shape, codebook_emb_p.shape, 'n', n, y_idx)
        z = torch.cat([codebook_emb_t.unsqueeze(1), codebook_emb_p], dim=1).to(device)
        result_dict = self.vq_vae.decoder(inputs=None, memory=z, generate=True, label_text=label_text)  # [batch_size, emb_size]
        return result_dict, y_idx, p_idx  # 截断梯度流

    def sample_all(self, y_idx, device, one_by_one=False, label_text=None):
        """
        Returns embedding tensor for a batch of indices.
        y_idx [1]
        """
        if self.vq_vae.p_vq_type in ['GS', 'DVQ'] and self.config.decompose_number > 1:
            result_dict, y_idx, p_idx = self.sample_result(y_idx=y_idx, device=device)
        else:
            codebook_emb_t, _ = self.sample_labels(y_idx=y_idx, n=0)  # [batch_size, distill_size]
            p_idx = torch.arange(0, self.disper_num, 1).to(device).unsqueeze(1)
            # print(p_idx)
            codebook_emb_p = self.vq_vae.emb_p.quantize_embedding(p_idx)  # [disper num, disperse_size]
            # result
            result = torch.cat([codebook_emb_t.unsqueeze(1).repeat(codebook_emb_p.shape[0], 1, 1),
                                codebook_emb_p], dim=1).to(device)
            if one_by_one is True:
                result_dict = {}
                for i in range(result.shape[0]):
                    one_result_dict = self.vq_vae.decoder(inputs=None, memory=result[i, :, :].unsqueeze(0), generate=True, label_text=label_text)  # [batch_size, emb_size]
                    if result_dict == {}:
                        result_dict['pred_idx'] = one_result_dict['pred_idx']
                    else:
                        result_dict['pred_idx'] = torch.cat([result_dict['pred_idx'], one_result_dict['pred_idx']], dim=0)
            else:
                result_dict = self.vq_vae.decoder(inputs=None, memory=result, generate=True, label_text=label_text)  # [batch_size, emb_size]
        return result_dict, y_idx, p_idx  # 截断梯度流
