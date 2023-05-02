import torch
import torch.nn as nn
import numpy as np


class BPNetwork(nn.Module):

    def __init__(self, n_in, n_hidden, n_out, 
                 activation='relu', output_activation='linear',
                 bias=True, initialization='orthogonal',
                 random_seed=None):
        super().__init__()

        self._n_in = n_in
        self._n_hidden = n_hidden
        self._n_out = n_out

        if random_seed is not None:
            torch.manual_seed(random_seed)

        if n_hidden is None:
            n_all = [n_in, n_out]
        else:
            n_all = [n_in] + n_hidden + [n_out]
        self.layers = nn.ModuleList()
        for i in range(1, len(n_all)):
            layer = nn.Linear(n_all[i-1], n_all[i], bias=bias)
            if initialization == 'orthogonal':
                gain = np.sqrt(6. / (n_all[i-1] + n_all[i]))
                nn.init.orthogonal_(layer.weight, gain=gain)
            elif initialization == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif initialization == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif initialization == 'teacher':
                nn.init.xavier_normal_(layer.weight, gain=3.)
            else:
                raise ValueError('Provided weight initialization "{}" is not '
                                 'supported.'.format(initialization))
            if bias:
                nn.init.constant_(layer.bias, 0)

            self.layers.append(layer)
        if n_hidden is not None:
            if isinstance(activation, str):
                self.activation = [activation]*len(n_hidden)
            else:
                self.activation = activation
        self.output_activation = output_activation

    @staticmethod
    def nonlinearity(x, nonlinearity):
        if nonlinearity == 'tanh':
            return torch.tanh(x)
        elif nonlinearity == 'relu':
            return F.relu(x)
        elif nonlinearity == 'cap_relu':
            return torch.clamp(x, 0, 1)
        elif nonlinearity == 'linear':
            return x
        elif nonlinearity == 'leakyrelu':
            return F.leaky_relu(x, 0.2)
        elif nonlinearity == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(nonlinearity))


    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.nonlinearity(x, self.activation[i])

        x = self.layers[-1](x)
        x = self.nonlinearity(x, self.output_activation)
        return x


    def save_logs(self, writer, step):

        for i, param in enumerate(self.parameters()):
            if i%2 == 0:
                param_name = 'weights'
            else:
                param_name = 'bias'

            forward_weights_norm = torch.norm(param)
            writer.add_scalar(tag='layer_{}/{}_norm'.format(int(i/2+1), param_name),
                              scalar_value=forward_weights_norm,
                              global_step=step)
            if param.grad is not None:
                forward_weights_gradients_norm = torch.norm(param.grad)
                writer.add_scalar(
                    tag='layer_{}/{}_gradients_norm'.format(int(i/2+1), param_name),
                    scalar_value=forward_weights_gradients_norm,
                    global_step=step)


    def set_requires_grad(self, value):
        """
        Sets the 'requires_grad' attribute of the all the parameters
        to the given value
        Args:
            value (bool): True or False
        """

        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')

        for param in self.parameters():
            param.requires_grad = value