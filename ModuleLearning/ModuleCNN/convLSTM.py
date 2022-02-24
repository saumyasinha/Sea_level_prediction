import torch.nn as nn
import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, saving_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.saving_path = saving_path

    def __call__(self, val_loss, model, epoch, parallel=False):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,parallel)
        elif (score < self.best_score + self.delta):
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        elif epoch>5: #20:
            self.best_score = score
            self.save_checkpoint(val_loss, model, parallel)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,parallel):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))
        if parallel==True:
            torch.save(model.module.state_dict(), self.saving_path)
        else:
            torch.save(model.state_dict(), self.saving_path)
        self.val_loss_min = val_loss


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dilation_rate =(1,2,4)):
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
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.dilation_rate = dilation_rate
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, dilation=self.dilation_rate[0], padding=self.padding, bias = self.bias),
            # nn.ReLU(),
            nn.Conv2d(4 * self.hidden_dim, 4 * self.hidden_dim, kernel_size=self.kernel_size, dilation=self.dilation_rate[1], padding=(2,2), bias=self.bias),
            # nn.ReLU(),
            nn.Conv2d(4 * self.hidden_dim, 4 * self.hidden_dim, kernel_size=self.kernel_size,
                      dilation=self.dilation_rate[2], padding=(4,4),
                      bias=self.bias),
            # nn.ReLU(),
        )
        #self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
           #                   out_channels=4 * self.hidden_dim,
          #                    kernel_size=self.kernel_size,
         #                     dilation = self.dilation_rate,
        #                      padding=self.padding,
       #                       bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

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
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device))


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
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, last_channel_size=1,
                 batch_first=False, bias=True, return_all_layers=False):
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
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(self.hidden_dim[-1], last_channel_size, 1)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()
        #print(b)
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        # layer_output_list = []
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

            if layer_idx == self.num_layers -1:
                h = self.conv_last(h)
                h = torch.squeeze(h)

            ## the layer output will still have the old h
            # layer_output_list.append(layer_output)
            last_state_list.append([h, c])


        if not self.return_all_layers:
            # layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return last_state_list


    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param




def trainBatchwise(trainX, trainY, validX,
                   validY, weight_map, train_mask, valid_mask, n_output_length, n_features, n_timesteps, epochs, batch_size, lr,
                   folder_saving, model_saved, quantile, alphas, outputs_quantile, valid, hidden_dim, num_layers, kernel_size, patience=None, verbose=None,
                   reg_lamdba=0):  # 0.0001):

    basic_forecaster = ConvLSTM(input_dim=n_features,
                                hidden_dim=hidden_dim,
                                kernel_size=kernel_size,
                                num_layers=num_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("num of parameters in this mode:", count_parameters(basic_forecaster))

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    parallel = False
    if train_on_gpu:

        if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            basic_forecaster = nn.DataParallel(basic_forecaster)
            parallel = True

        basic_forecaster = basic_forecaster.cuda()
        weight_map = weight_map.cuda()

    print(basic_forecaster)

    optimizer = torch.optim.Adam(basic_forecaster.parameters(), lr=lr) #, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    # criterion = torch.nn.MSELoss()
    # criterion = nn.L1Loss()
    samples = trainX.size()[0]
    losses = []
    valid_losses = []

    saving_path = folder_saving + model_saved
    early_stopping = EarlyStopping(saving_path, patience=patience, verbose=True)
    train_mode = False

    for epoch in range(epochs):
        if train_mode is not True:
            basic_forecaster.train()
            train_mode = True

        indices = torch.randperm(samples)
        trainX, trainY, train_mask = trainX[indices, :, :, :, :], trainY[indices, :, :], train_mask[indices, :, :]
        per_epoch_loss = 0
        count_train = 0
        for i in range(0, samples, batch_size):
            xx = trainX[i: i + batch_size, :, :, :, :]
            yy = trainY[i: i + batch_size, :, :]
            batch_mask = train_mask[i: i + batch_size, :, :]
            # batch_weight_map = weight_map_train[i: i + batch_size, :, :]

            if train_on_gpu:
                xx, yy = xx.cuda(), yy.cuda()

            outputs = basic_forecaster.forward(xx)[0][0]
            optimizer.zero_grad()
            if quantile:
                loss = quantile_loss(outputs, yy, alphas, batch_mask, weight_map)

            else:
                loss = MaskedMSELoss(outputs, yy, batch_mask, weight_map)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            #   scheduler.step()
            per_epoch_loss += loss.item()
            count_train += 1

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

        train_loss_this_epoch = per_epoch_loss / count_train
        losses.append(train_loss_this_epoch)

        if valid:
            train_mode = False
            basic_forecaster.eval()

            samples_valid = validX.size()[0]
            count_valid = 0
            valid_loss = 0
            for i in range(0, samples_valid, batch_size):
                xx_valid = validX[i: i + batch_size, :, :, :, :]
                yy_valid = validY[i: i + batch_size, :, :]
                batch_mask_valid = valid_mask[i: i + batch_size, :, :]
                # batch_weight_map_valid = weight_map_valid[i: i + batch_size, :, :]

                if train_on_gpu:
                    xx_valid, yy_valid = xx_valid.cuda(), yy_valid.cuda()
                    # validX,validY,weight_map_valid = validX.cuda(), validY.cuda(),weight_map_valid.cuda()

                validYPred = basic_forecaster.forward(xx_valid)[0][0]
                if quantile:
                    valid_loss += quantile_loss(validYPred, yy_valid, alphas, batch_mask_valid, weight_map).item()

                    # valid_loss = self.crps_score(validYPred, validYTrue, np.arange(0.05, 1.0, 0.05))s
                    # valid_losses.append(valid_loss_this_epoch)
                    # print("Epoch: %d, loss: %1.5f and valid_loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))
                else:
                    valid_loss += MaskedMSELoss(validYPred, yy_valid, batch_mask_valid, weight_map).item()

                count_valid += 1

            valid_loss_this_epoch = valid_loss / count_valid
            valid_losses.append(valid_loss_this_epoch)
            print("Epoch: %d, train loss: %1.5f and valid loss : %1.5f" % (
            epoch, train_loss_this_epoch, valid_loss_this_epoch))

            # early_stopping(valid_loss, self)
            early_stopping(valid_loss_this_epoch, basic_forecaster, epoch, parallel)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            print("Epoch: %d, loss: %1.5f" % (epoch, train_loss_this_epoch))
            if parallel:
                torch.save(basic_forecaster.module.state_dict(), saving_path)
            else:
                torch.save(basic_forecaster.state_dict(), saving_path)
    return losses, valid_losses


def quantile_loss(outputs, target, alphas, mask):
    ## modify this loss for the new data, also apply mask correctly

    for i, alpha in zip(range(len(alphas)), alphas):
        output = outputs[:, i].reshape((-1, 1))
        covered_flag = (output <= target).float()
        uncovered_flag = (output > target).float()
        if i == 0:
            loss = ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)
        else:
            loss += ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)

    return torch.mean(loss)


def MaskedMSELoss(pred, target, mask, weight_map):
    # print(mask.requires_grad)
    # mask = mask.detach()

    weight_map = weight_map.unsqueeze(0).repeat(len(target), 1, 1)

    diff = (target - pred)
    weighted_diff2 = (diff ** 2) * weight_map  # .expand_as(diff)
    weighted_diff2 = weighted_diff2[mask]
    weights_masked = weight_map[mask]
    # loss = weighted_diff2.mean()
    loss = weighted_diff2.sum() / weights_masked.sum()
    # print(loss)
    return loss
