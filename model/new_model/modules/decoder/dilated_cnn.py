import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DilatedCNNLayer(nn.Module):
    def __init__(self, externel_size, internel_size, dilation_size, dropout_rate, device):
        super(DilatedCNNLayer, self).__init__()
        self.externel_size = externel_size
        self.internel_size = internel_size
        self.dilation_size = dilation_size
        self.pad_size = 2 * self.dilation_size
        self.dropout_rate = dropout_rate
        self.device = device
        # Layers
        # dilated CNN 1
        self.dcnn_layer1 = nn.Conv2d(in_channels=self.externel_size,
                                     out_channels=self.internel_size,
                                     kernel_size=(1, 1),
                                     bias=False)
        nn.init.kaiming_normal_(self.dcnn_layer1.weight)
        self.batch_norm1 = nn.BatchNorm2d(self.internel_size,
                                          affine=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # dilated CNN 2
        self.dcnn_layer2 = nn.Conv2d(in_channels=self.internel_size,
                                     out_channels=self.internel_size,
                                     kernel_size=(1, 3),
                                     dilation=(1, self.dilation_size),
                                     bias=False)
        nn.init.kaiming_normal_(self.dcnn_layer2.weight)
        self.batch_norm2 = nn.BatchNorm2d(self.internel_size,
                                          affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # dilated CNN 3
        self.dcnn_layer3 = nn.Conv2d(in_channels=self.internel_size,
                                     out_channels=self.externel_size,
                                     kernel_size=(1, 1),
                                     bias=False)
        nn.init.kaiming_normal_(self.dcnn_layer3.weight)
        self.batch_norm3 = nn.BatchNorm2d(self.externel_size,
                                          affine=True)
        self.relu3 = nn.ReLU()

    def forward(self, x, is_training=True):
        '''
        x: # [batch, cnn_exter, 1, seq_len - 1]
        '''
        # first cnn layer
        cnn_out1 = self.dcnn_layer1(x)  # [batch, cnn_inter, 1, seq_len-1]
        cnn_out1 = self.relu1(self.batch_norm1(cnn_out1))  # [batch, cnn_inter, 1, seq_len - 1]
        cnn_out1 = self.dropout1(cnn_out1)
        pad = torch.zeros([cnn_out1.shape[0], cnn_out1.shape[1], 1, self.pad_size]).to(x.device)
        cnn_out1 = torch.cat([pad, cnn_out1, pad], dim=3)  # [batch, cnn_inter, 1, seq_len - 1 + 2 * pad_size]
        # second cnn layer
        cnn_out2 = self.dcnn_layer2(cnn_out1)  # [batch, cnn_inter, 1, seq_len-1 + pad_size]
        cnn_out2 = cnn_out2[:, :, :, :-self.pad_size]  # [batch, cnn_inter, 1, seq_len - 1]
        cnn_out2 = self.relu2(self.batch_norm2(cnn_out2))  # [batch, cnn_inter, 1, seq_len - 1]
        cnn_out2 = self.dropout2(cnn_out2)
        # third cnn layer
        cnn_out3 = self.dcnn_layer3(cnn_out2)  # [batch, cnn_exter, 1, seq_len-1 + pad_size]
        cnn_out3 = self.batch_norm3(cnn_out3)  # [batch, cnn_exter, 1, seq_len - 1]
        # resnet
        result = self.relu3(cnn_out3 + x)  # [batch, cnn_exter, 1, seq_len - 1]
        return result


class DilatedCNN(nn.Module):
    def __init__(self, config, device, decoder_name='dilatedcnn', bos=None):
        super(DilatedCNN, self).__init__()
        self.input_size = config.distil_size + config.disper_size
        self.emb_size = config.emb_size
        self.cnn_internel_size = config.cnn_internel_size
        self.dropout_rate = config.dropout_rate
        self.seq_len = config.seq_len
        self.output_size = config.emb_size
        if bos is None:
            self.bos = torch.zeros([1, 1, self.emb_size])  # [1, 1, emb_size]
        else:
            self.bos = bos
        

        # Layers
        # dilated CNN 1
        self.cnn_layer1 = DilatedCNNLayer(externel_size=self.input_size + self.emb_size,
                                          internel_size=self.cnn_internel_size,
                                          dilation_size=1,
                                          dropout_rate=self.dropout_rate,
                                          device=device)
        # dilated CNN 2
        self.cnn_layer2 = DilatedCNNLayer(externel_size=self.input_size + self.emb_size,
                                          internel_size=self.cnn_internel_size,
                                          dilation_size=2,
                                          dropout_rate=self.dropout_rate,
                                          device=device)
        # dilated CNN 3
        self.cnn_layer3 = DilatedCNNLayer(externel_size=self.input_size + self.emb_size,
                                          internel_size=self.cnn_internel_size,
                                          dilation_size=4,
                                          dropout_rate=self.dropout_rate,
                                          device=device)
        # output layer
        self.output_layer = nn.Linear(self.input_size + self.emb_size, self.emb_size, bias=True)

    def forward(self, x, z, pad_mask=None, is_training=True):
        '''
        x: [batch_size, seq_len, emb_size]
        z: [batch_size, latent_size]
        output: [batch_size, seq_len, emb_size]
        '''
        if is_training and self.training:
            z = z.unsqueeze(1)
            z_matrix = z.repeat(1, self.seq_len - 1, 1)  # expand to [batch_size, seq_len-1, latent_size]
            bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(x.device)  # expand to [batch_size, 1, emb_size]
            x_matrix = x[:, 1:-1, :]  # [batch_size, seq_len-2, emb_size]
            x_matrix = torch.cat([bos_matrix, x_matrix], dim=1)  # [batch_size, seq_len-1, emb_size]
            cnn_input = torch.cat([x_matrix, z_matrix], dim=2)  # [batch_size, seq_len-1, latent_size+emb_size]
            cnn_input = cnn_input.permute(0, 2, 1)  # [batch, latent_size+emb_size, seq_len-1]
            cnn_input = cnn_input.unsqueeze(2)  # [batch, latent_size+emb_size, 1, seq_len-1]
            output = self.cnn_layer1(cnn_input)  # [batch, latent_size+emb_size, 1, seq_len-1]
            output = self.cnn_layer2(output)  # [batch, latent_size+emb_size, 1, seq_len-1]
            output = self.cnn_layer3(output)  # [batch, latent_size+emb_size, 1, seq_len-1]
            output = output.squeeze(2)  # [batch, latent_size+emb_size, seq_len-1]
            output = output.permute(0, 2, 1)  # [batch, seq_len-1, latent_size+emb_size]
            result = self.output_layer(output)   # [batch, seq_len-1, emb_size]
        else:
            z = z.unsqueeze(1)  # [batch_size, 1, latent_size]
            bos_matrix = self.bos.repeat(z.shape[0], 1, 1).to(z.device)  # expand to [batch_size, 1, emb_size]
            #next_input = torch.cat([bos_matrix, z], dim=2)  # [batch_size, 1, emb_size+latent_size]
            next_input = bos_matrix
            result = []
            for i in range(self.seq_len-1):
                next_input = torch.cat([next_input, z], dim=2)  # [batch_size, 1, emb_size+latent_size]
                next_input = next_input.permute(0, 2, 1)  # [batch, latent_size+emb_size, 1]
                next_input = next_input.unsqueeze(2)  # [batch, latent_size+emb_size, 1, 1]
                output = self.cnn_layer1(next_input)  # [batch, latent_size+emb_size, 1, 1]
                output = self.cnn_layer2(output)  # [batch, latent_size+emb_size, 1, 1]
                output = self.cnn_layer3(output)  # [batch, latent_size+emb_size, 1, 1]
                output = output.squeeze(2)  # [batch, latent_size+emb_size, 1]
                next_input = output.permute(0, 2, 1)  # [batch, 1, latent_size+emb_size]
                next_input = self.output_layer(next_input)  # [batch, 1, emb_size]
                result.append(next_input)
            result = torch.cat(result, dim=1)  # [batch, seq_len-1, latent_size+emb_size]
        result = torch.cat([bos_matrix, result], dim=1)  # [batch, seq_len, emb_size]
        return result
