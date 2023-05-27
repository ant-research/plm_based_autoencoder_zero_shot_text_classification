from unittest import result
import torch
import torch.nn as nn


class LinearDiscriminator(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        input_dim = config.encoder_output_size
        self.ln_1 = nn.Linear(input_dim, input_dim, bias=True)
        self.ln_2 = nn.Linear(input_dim, 1, bias=True)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, label: torch.Tensor):
        """
        inputs: (2d tensor), batch of vector
        """
        with torch.cuda.amp.autocast():
            hid = torch.tanh(self.ln_1(inputs))
            output = self.ln_2(hid)
            # for i in range(output.shape[0]):
            #     print('output is', output[i, :], inputs[i, :], label[i])
            # print('probability output is', output)

            loss = self.loss_function(inputs=output, targets=label.to(output.device))
            result_dict = {
                'prob': output,
                'loss': loss
            }
            return result_dict

    def loss_function(self, inputs, targets):
        return self.loss_func(inputs.squeeze(1), targets.float())


class LinearDisentangleDiscriminator(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        input_dim = config.class_num + config.disper_size * 2
        self.ln_1 = nn.Linear(input_dim, input_dim, bias=True)
        self.ln_2 = nn.Linear(input_dim, 1, bias=True)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, label: torch.Tensor):
        """
        inputs: (2d tensor), batch of vector
        """
        with torch.cuda.amp.autocast():
            hid = torch.tanh(self.ln_1(inputs))
            output = self.ln_2(hid)
            # for i in range(output.shape[0]):
            #     print('output is', output[i, :], inputs[i, :], label[i])
            # print('probability output is', output)

            loss = self.loss_function(inputs=output, targets=label.to(output.device))
            result_dict = {
                'prob': output,
                'loss': loss
            }
            return result_dict

    def loss_function(self, inputs, targets):
        return self.loss_func(inputs.squeeze(1), targets.float())


Discriminator = {
    'Linear': LinearDisentangleDiscriminator
}
