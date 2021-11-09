import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# from Sea_level_prediction.ModuleLearning.ModuleCNN.unet import UNet
from ModuleLearning.ModuleCNN.unet import UNet



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
    def __init__(self, quantile, outputs, dim_channels, dim_h=360, dim_w=180):
        super(FullyConvNet, self).__init__()
        # Input size: [batch, 13, 360, 180]

        if quantile:
            last_channel_size = outputs
        else:
            last_channel_size = 1

    #     # smaller FCN
     #   self.encoder = nn.Sequential(
     #        nn.Conv2d(dim_channels, 16, (3,3), stride=(2,2), padding = (1,1)),  #45*23
      #      # nn.BatchNorm2d(16),
      #       nn.ReLU(),
       #      nn.Conv2d(16, 32,(3,3),stride=(2,2),  padding = (1,1)),#23*12
         #   # nn.BatchNorm2d(32),
        #     nn.ReLU(),
          # #  nn.Dropout(0.3),
         #    nn.Conv2d(32, 64, (3,3), stride =(2,2), padding = (1,1)), #12*6
         # #   nn.BatchNorm2d(64),
          #   nn.ReLU())
    #     #   # #nn.Dropout(0.3))
    
       # self.decoder = nn.Sequential(
    #
     #        nn.ConvTranspose2d(64, 32, (3,3), stride = (2,2), padding = (0,1)), #23*12
             # nn.BatchNorm2d(32),
      #       nn.ReLU(),
             # nn.Dropout(0.3),
       #      nn.ConvTranspose2d(32, 16,(3,3), stride=(2,2), padding = (1,1)), #45*23
             # nn.BatchNorm2d(16),
        #     nn.ReLU(),
    #         ##nn.Dropout(0.3),
         #    nn.ConvTranspose2d(16, last_channel_size, (2, 4), stride=(2,2), padding = (1,0))) #90*45
    # #
        # ## SMALL fcn version2
      #  self.encoder = nn.Sequential(
      #   nn.Conv2d(dim_channels, 32, (3, 3), stride=2, padding=1),  # 45*23
    # #     #     nn.BatchNorm2d(32),
      #  nn.ReLU(),
      #   nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),  # 23*12
    # #     #     nn.BatchNorm2d(64),
      #  nn.ReLU(),
      #   nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),  # 12*6
    # #     #     nn.BatchNorm2d(128),
      #  nn.ReLU(),
    # #     #     nn.Dropout(0.3)
    # )
    # #     #
     #   self.decoder = nn.Sequential(
      #  nn.ConvTranspose2d(128, 64, (3,2), stride = 2, padding = (1,0)), #23*12
    # #     #     nn.BatchNorm2d(64),
      #   nn.ReLU(),
    # #     #     nn.Dropout(0.3),
       #  nn.ConvTranspose2d(64, 32,(3,3), stride=2, padding = 1), #45*23
    # #     #     nn.BatchNorm2d(32),
       #  nn.ReLU(),
       #  nn.ConvTranspose2d(32, last_channel_size, (2, 3), stride=2, padding = (0,1)) #90*45
    # #     #
    # #     #     # nn.Sigmoid(),
    # )
    # #     #
    # #
     #   self.apply(weights_init)
    # #
    # #
    # #
    #def forward(self, x):
     #   x = self.encoder(x)
         # print(x.shape)
      #  x = self.decoder(x)
    #     print(x.shape)
       # x = torch.squeeze(x)
    #    # # print(x.shape)
       # return x


        self.model = UNet(dim_channels,last_channel_size)
   # #
    def forward(self, x):
        x = self.model(x)
        x = torch.squeeze(x)
      #  # print(x.shape)
        return x


def trainBatchwise(trainX, trainY, validX,
                   validY, train_mask, valid_mask, n_output_length, n_features, n_timesteps,  epochs, batch_size, lr, folder_saving, model_saved, quantile, alphas, outputs_quantile, valid, patience=None, verbose=None, reg_lamdba = 0): #0.0001):

    basic_forecaster = FullyConvNet(quantile, outputs_quantile, n_timesteps)

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    parallel = False
    if train_on_gpu:
        if torch.cuda.device_count() > 1:
           # print("Let's use", torch.cuda.device_count(), "GPUs!")
            basic_forecaster = nn.DataParallel(basic_forecaster)
            parallel = True



        basic_forecaster = basic_forecaster.cuda()


    print(basic_forecaster)

    optimizer = torch.optim.Adam(basic_forecaster.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay = 1e-5)
    # scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
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
        trainX, trainY, train_mask = trainX[indices, :, :, :], trainY[indices, :, :], train_mask[indices, :, :]
        per_epoch_loss = 0
        count_train = 0
        for i in range(0, samples, batch_size):
            xx = trainX[i: i + batch_size, :, :, :]
            yy = trainY[i: i + batch_size, :, :]
            batch_mask = train_mask[i: i + batch_size, :, :]

            if train_on_gpu:
                xx, yy = xx.cuda(), yy.cuda()

            outputs = basic_forecaster.forward(xx)
            optimizer.zero_grad()
            if quantile:
                loss = quantile_loss(outputs, yy,alphas, batch_mask)

            else:
                loss = MaskedMSELoss(outputs, yy, batch_mask)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # scheduler.step()
            per_epoch_loss+=loss.item()
            count_train+=1


        train_loss_this_epoch = per_epoch_loss/count_train
        losses.append(train_loss_this_epoch)

        if valid:
            train_mode = False
            basic_forecaster.eval()
            if train_on_gpu:

                validX,validY = validX.cuda(), validY.cuda()


            if quantile:
                validYPred = basic_forecaster.forward(validX)
                valid_loss_this_epoch = quantile_loss(validYPred,validY,alphas,valid_mask).item()

                # valid_loss = self.crps_score(validYPred, validYTrue, np.arange(0.05, 1.0, 0.05))s
                valid_losses.append(valid_loss_this_epoch)
                print("Epoch: %d, loss: %1.5f and valid_loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))
            else:
                validYPred = basic_forecaster.forward(validX)
                valid_loss_this_epoch = MaskedMSELoss(validYPred, validY, valid_mask).item()
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



def MaskedMSELoss(pred, target, mask):
    # print(mask.requires_grad)
    # mask = mask.detach()
    diff = target - pred
    diff = diff[mask]
    loss = (diff ** 2).mean()
    return loss



def  MaskedL1Loss(pred, target, mask):
    #print(pred.shape, target.shape, mask.shape)
    # mask = mask.detach()
    diff = target - pred
    diff = diff[mask]
    loss = diff.abs().mean()
    return loss

def MaskedBerhuLoss(pred, target, mask):
    diff = target - pred
    diff = diff[mask]
    diff = diff.abs()
    # print(diff.max(),diff.mean())
    delta_mask = diff > 1

    loss = torch.mean((0.5 * delta_mask * (diff** 2)) + ~delta_mask * diff)
    return loss

