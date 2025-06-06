import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2, kernel_size[1]//2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
    # hidden_dim is the sequence length

class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        image_size: (int, int) Size of image
    
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W

    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - First list is layer outputs at the last time step
            1 - Second list is tuple of last hidden state and last cell state    
    
    Example:
        >> x = torch.rand((16, 8, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, (128, 128))
        >> y = convlstm(x)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters:
            input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            hidden_state: None. Used for prediction
        Returns:
            last_state_list, layer_output
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []        

        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    #  Every element of cell list has lstm cells equal to the number of time instances

    def _check_kernel_size_consistency(self, kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
        
    def _extend_for_multilayer(self, param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
# To extend a parameter for all the layers.


class ConvLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, batch_first,bias, return_all_layers):
        super().__init__()
        # ConvLSTM backbone
        self.conv_lstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=len(hidden_dims),
            batch_first=batch_first,
            bias=bias,
            return_all_layers=return_all_layers
        )
        # Final convolution to reduce channels to 1
        self.final_conv = nn.Conv2d(
            in_channels=hidden_dims[-1], 
            # Because the last hidden dimension will enter the lstm network.
            out_channels=1, 
            kernel_size=1  # 1x1 convolution
        )

    def forward(self, x):
        # x shape: (B, 16, 16, 480, 480) = (batch, time, channels, H, W)
        outputs, _ = self.conv_lstm(x)
        # print("Conv lstm outputs")
        # print(outputs)
        # print("Shape of Outputs")
        # print(outputs.shape)

        # outputs[0] shape: (B, 16, 64, 480, 480) = (batch, time, hidden_dim, H, W)
        

        # Apply final conv to reduce channels from 64 to 1
        b, t, c, h, w = outputs[0].shape
        # outputs is an array under a array
        out = outputs[0].view(b * t, c, h, w)  # (B*T, 64, 480, 480)
        
        # This merges the batch and time dimensions so that the tensor is reshaped into: (B∗T,64,480,480)
        #The reason for this is that standard 2D convolutions (nn.Conv2d) operate on 4D tensors with shape 
        #(N,C,H,W) where N is the number of images.

        out = self.final_conv(out)  # (B*T, 1, 480, 480)
        out = out.view(b, t, 1, h, w)  # (B, 16, 1, 480, 480)
        return out