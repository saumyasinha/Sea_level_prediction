import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# from Sea_level_prediction.ModuleLearning.ModuleCNN.unet import UNet
from ModuleLearning.ModuleCNN.unet import UNet, UNet_model,SmaAt_UNet_model, UNet_attn_model,Dilated_UNet_model,Dilated_UNet_attn_model, UNet3d_model, Dilated_UNet3d_model


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



# nn_initalization = nn.init.xavier_uniform
def weights_init(m):

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=1)
        # nn_initalization(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1)
        # nn_initalization(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)





class FullyConvNet(nn.Module):
    def __init__(self, model_type, quantile, outputs, dim_channels, dim_h=360, dim_w=180):
        super(FullyConvNet, self).__init__()
        # Input size: [batch, 13, 360, 180]

        if quantile:
            last_channel_size = outputs
        else:
            last_channel_size = 1


    # #     # smaller FCN
    # #     self.encoder = nn.Sequential(
    # #         nn.Conv2d(dim_channels, 16, (3,3), stride=(2,2), padding = (1,1)),  #45*23
    # #        # nn.BatchNorm2d(16),
    # #         nn.ReLU(),
    # #         nn.Conv2d(16, 32,(3,3),stride=(2,2),  padding = (1,1)),#23*12
    # #        # nn.BatchNorm2d(32),
    # #         nn.ReLU(),
    # #       #  nn.Dropout(0.3),
    # #         nn.Conv2d(32, 64, (3,3), stride =(2,2), padding = (1,1)), #12*6
    # #      #   nn.BatchNorm2d(64),
    # #         nn.ReLU())
    # #        # #nn.Dropout(0.3))
    # #
    # #     self.decoder = nn.Sequential(
    # #
    # #         nn.ConvTranspose2d(64, 32, (3,3), stride = (2,2), padding = (0,1)), #23*12
    # #         # nn.BatchNorm2d(32),
    # #         nn.ReLU(),
    # #         # nn.Dropout(0.3),
    # #         nn.ConvTranspose2d(32, 16,(3,3), stride=(2,2), padding = (1,1)), #45*23
    # #         # nn.BatchNorm2d(16),
    # #         nn.ReLU(),
    # #         ##nn.Dropout(0.3),
    # #         nn.ConvTranspose2d(16, last_channel_size, (2, 4), stride=(2,2), padding = (1,0))) #90*45
    #
    #     ## SMALL fcn version2
    #     self.encoder = nn.Sequential(
    #         nn.Conv2d(dim_channels, 32, (3, 3), stride=2, padding=1),  # 45*23
    #         # nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),  # 23*12
    #         # nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),  # 12*6
    #         # nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         # nn.Dropout(0.3)
    #     )
    #
    #     self.decoder = nn.Sequential(
    #         nn.ConvTranspose2d(128, 64, (3,2), stride = 2, padding = (1,0)), #23*12
    #         # nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         # nn.Dropout(0.3),
    #         nn.ConvTranspose2d(64, 32,(3,3), stride=2, padding = 1), #45*23
    #         # nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(32, last_channel_size, (2, 3), stride=2, padding = (0,1)) #90*45
    #
    #         # nn.Sigmoid(),
    #     )
    #
    #
    #     self.apply(weights_init)
    # #
    # #
    # #
    # def forward(self, x):
    #     x = self.encoder(x)
    #     # print(x.shape)
    #     x = self.decoder(x)
    #     # print(x.shape)
    #     x = torch.squeeze(x)


        if model_type == "Unet3d":
            self.model = UNet3d_model(dim_channels=1,last_channel_size=last_channel_size)
        if model_type == "DilatedUnet3d":
            self.model = Dilated_UNet3d_model(dim_channels=1,last_channel_size=last_channel_size)
        if model_type == "Unet":
            self.model = UNet_model(dim_channels,last_channel_size)
        if model_type == "DilatedUnet":
            self.model = Dilated_UNet_model(dim_channels, last_channel_size)
        if model_type == "SmaAT_Unet":
            self.model = SmaAt_UNet_model(dim_channels, last_channel_size)
        if model_type == "Unet_Attn":
            self.model = UNet_attn_model(dim_channels, last_channel_size)
        if model_type == "DilatedUnet_Attn":
            self.model = Dilated_UNet_attn_model(dim_channels, last_channel_size)

   # #
    def forward(self, x):
        x = self.model(x)
        x = torch.squeeze(x)
      #  # print(x.shape)
        return x


def trainBatchwise(model_type, trainX, trainY, validX,
                   validY,  weight_map, train_mask, valid_mask, n_output_length, n_features, n_timesteps,  epochs, batch_size, lr, folder_saving, model_saved, quantile, alphas, outputs_quantile, valid, patience=None, verbose=None, reg_lamdba = 0): #0.0001):


    basic_forecaster = FullyConvNet(model_type, quantile, outputs_quantile, n_timesteps)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of parameters in this model:",count_parameters(basic_forecaster))


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

    optimizer = torch.optim.Adam(basic_forecaster.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay = 1e-5)
    #optimizer = torch.optim.SGD(basic_forecaster.parameters(), lr=lr)
    # samples = trainX.size()[0]
    #steps = int(samples/batch_size)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    # criterion = torch.nn.MSELoss()
    # criterion = nn.L1Loss()
    samples = trainX.size()[0]
    losses = []
    valid_losses = []

    saving_path = folder_saving+model_saved
    early_stopping = EarlyStopping(saving_path, patience=patience, verbose=True)
    train_mode = False


    for epoch in range(epochs):
        if train_mode is not True:
            basic_forecaster.train()
            train_mode = True

        indices = torch.randperm(samples)
        trainX, trainY, train_mask= trainX[indices, :, :, :], trainY[indices, :, :], train_mask[indices, :, :]
        per_epoch_loss = 0
        count_train = 0
        for i in range(0, samples, batch_size):
            xx = trainX[i: i + batch_size, :, :, :]
            yy = trainY[i: i + batch_size, :, :]
            batch_mask = train_mask[i: i + batch_size, :, :]
            # batch_weight_map = weight_map_train[i: i + batch_size, :, :]

            if train_on_gpu:

                xx, yy = xx.cuda(), yy.cuda()

            outputs = basic_forecaster.forward(xx)
            optimizer.zero_grad()
            if quantile:
                loss = quantile_loss(outputs, yy,alphas, batch_mask, weight_map)

            else:
                loss = MaskedMSELoss(outputs, yy, batch_mask, weight_map)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
         #   scheduler.step()
            per_epoch_loss+=loss.item()
            count_train+=1

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)


        train_loss_this_epoch = per_epoch_loss/count_train
        losses.append(train_loss_this_epoch)

        if valid:
            train_mode = False
            basic_forecaster.eval()

            samples_valid = validX.size()[0]
            count_valid = 0
            valid_loss = 0
            for i in range(0, samples_valid, batch_size):
                xx_valid = validX[i: i + batch_size, :, :, :]
                yy_valid = validY[i: i + batch_size, :, :]
                batch_mask_valid = valid_mask[i: i + batch_size, :, :]
                # batch_weight_map_valid = weight_map_valid[i: i + batch_size, :, :]

                if train_on_gpu:
                    xx_valid, yy_valid = xx_valid.cuda(), yy_valid.cuda()
                    #validX,validY,weight_map_valid = validX.cuda(), validY.cuda(),weight_map_valid.cuda()

                validYPred = basic_forecaster.forward(xx_valid)
                if quantile:
                    valid_loss +=quantile_loss(validYPred,yy_valid,alphas,batch_mask_valid,weight_map).item()

                    # valid_loss = self.crps_score(validYPred, validYTrue, np.arange(0.05, 1.0, 0.05))s
                    #valid_losses.append(valid_loss_this_epoch)
                    #print("Epoch: %d, loss: %1.5f and valid_loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))
                else:
                    valid_loss += MaskedMSELoss(validYPred, yy_valid, batch_mask_valid, weight_map).item()

                count_valid+=1

            valid_loss_this_epoch = valid_loss/count_valid
            valid_losses.append(valid_loss_this_epoch)
            print("Epoch: %d, train loss: %1.5f and valid loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))

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

    for i, alpha in zip(range(len(alphas)),alphas):
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

    weight_map = weight_map.unsqueeze(0).repeat(len(target),1,1)

    diff = (target - pred)
    weighted_diff2 = (diff ** 2)*weight_map #.expand_as(diff)
    weighted_diff2 = weighted_diff2[mask]
    weights_masked = weight_map[mask]
    # loss = weighted_diff2.mean()
    loss = weighted_diff2.sum()/weights_masked.sum()
    # print(loss)
    return loss



# def  MaskedL1Loss(pred, target, mask):
#     #print(pred.shape, target.shape, mask.shape)
#     # mask = mask.detach()
#     diff = target - pred
#     diff = diff[mask]
#     loss = diff.abs().mean()
#     return loss
#
# def MaskedBerhuLoss(pred, target, mask):
#     diff = target - pred
#     diff = diff[mask]
#     diff = diff.abs()
#     # print(diff.max(),diff.mean())
#     delta_mask = diff > 1
#
#     loss = torch.mean((0.5 * delta_mask * (diff** 2)) + ~delta_mask * diff)
#     return loss
#
#
#
# def MaskedHuberLoss(pred, target, mask):
#     diff = target - pred
#     diff = diff[mask]
#     diff = diff.abs()
#     delta_mask = diff < 1
#
#     loss = torch.mean((0.5 * delta_mask * (diff** 2)) + ~delta_mask * diff)
#     return loss
#
