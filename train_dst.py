"""Definition of the training and validation loops"""
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert torch.cuda.is_available()
from torchmetrics.classification import Dice, MulticlassAccuracy
from model3d import Unet3D

from dataset_dst import (
    BlastoDataset
)

# Create training dataset
train_data = BlastoDataset("/group/dl4miacourse/projects/BlastoSeg/training")
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8)

val_data = BlastoDataset("/group/dl4miacourse/projects/BlastoSeg/validation")
val_loader = DataLoader(val_data, batch_size = 1, shuffle=False, num_workers=8)

# unet = UNet3D(depth=4, in_channels=1, out_channels=1, num_fmaps=2).to(device)
unet = Unet3D(n_classes = 2)
optimizer = torch.optim.Adam(unet.parameters()) # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 

def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
    batchsize = 5,
    step_size = 2
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x_batch, y_batch, loss_mask) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x_batch, y_batch, loss_mask = x_batch.to(device), y_batch.to(device), loss_mask.to(device)
        # Expected dimensions: B, C, D, H, W. In our case this should be 5, 1, D, 256, 256

        z_slices = x_batch.shape[1]
        num_iterations = int(z_slices / step_size)
        loss = 0
        for i in range(0, num_iterations, step_size): 

            start_index = i
            end_index = min(start_index + step_size, x_batch.size(2))

            x = x_batch[:, start_index:end_index, :, :] # [1,2,256,256] BDHW
            x = torch.unsqueeze(x, 0) # [1,1,2,256,256] BCDHW

            y = y_batch[:, start_index:end_index, :, :]
            y = torch.unsqueeze(y, 0)

            lm = loss_mask[:, start_index:end_index, :, :]
            lm = torch.unsqueeze(lm, 0)

            prediction = model(x)  # Assuming model expects a batch dimension
       
            optimizer.zero_grad()

            # apply model and calculate loss
            prediction = model(x)  # Assuming model expects a batch dimension

            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            loss = loss_function(prediction, y)
            loss = loss * lm
            loss = loss.mean()

            # backpropagate the loss and adjust the parameters
            loss.backward()
            optimizer.step()

            # log to console
            if batch_id % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_id * len(x_batch) + i,  # Adjusted index calculation
                        len(loader.dataset),
                        100.0 * batch_id / len(loader),
                        loss.item(),
                    )
                )
            # log to tensorboard
            if tb_logger is not None:
                step = epoch * len(loader.dataset) + batch_id * len(x_batch) + i  # Adjusted step calculation
                tb_logger.add_scalar(
                    tag="train_loss", scalar_value=loss.item(), global_step=step
                )
                # # check if we log images in this iteration
                # if step % log_image_interval == 0:
                #     tb_logger.add_images(
                #         tag="input", img_tensor=x.to("cpu"), global_step=step  # Assuming input requires batch dimension
                #     )
                #     tb_logger.add_images(
                #         tag="target", img_tensor=y.to("cpu"), global_step=step  # Assuming target requires batch dimension
                #     )
                #     tb_logger.add_images(
                #         tag="prediction",
                #         img_tensor=prediction.to("cpu").detach(),
                #         global_step=step,
                #     )
            if early_stop and batch_id > 5:
                print("Stopping test early!")
                break

def validate(
    model,
    loader,
    loss_function,
    batchsize,
    step=None,
    tb_logger=None,
    device=None,
    step_size = 2
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x_batch, y_batch, loss_mask in loader:
            # move input and target to the active device (either cpu or gpu)
            x_batch, y_batch, loss_mask = x_batch.to(device), y_batch.to(device), loss_mask.to(device)

            # Loop over each slice in the batch

            z_slices = x_batch.shape[1]
            num_iterations = int(z_slices / step_size)
            loss = 0
            for i in range(0,num_iterations,step_size): 

                start_index = i
                end_index = min(start_index + step_size, x_batch.size(2))

                x = x_batch[:, start_index:end_index, :, :] # [1,2,256,256] BDHW
                x = torch.unsqueeze(x, 0) # [1,1,2,256,256] BCDHW

                y = y_batch[:, start_index:end_index, :, :]
                y = torch.unsqueeze(y, 0)

                lm = loss_mask[:, start_index:end_index, :, :]
                lm = torch.unsqueeze(lm, 0)

                prediction = model(x)  # Assuming model expects a batch dimension
        
                if y.dtype != prediction.dtype:
                    y = y.type(prediction.dtype)
                loss = loss_function(prediction, y)
                loss = loss * lm
                loss = loss.mean()            
                val_loss += loss

    # normalize loss and metric
    val_loss /= len(loader)

    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        # # we always log the last validation images
        # tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        # tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
        # tb_logger.add_images(
        #     tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
        # )

    print(
        "\nValidate: Average loss: {:.4f}\n".format(
            val_loss
        )
    )

    return val_loss

class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # compluted as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        union = (prediction * prediction).sum() + (target * target).sum()
        return 2 * intersection / union.clamp(min=self.eps)

dice = DiceCoefficient()
loss = nn.MSELoss(reduction ='none')

logger = SummaryWriter("runs/UNet_MSEloss_stepsize2")
n_epochs = 40
best_val_loss = 1000
for epoch in range(n_epochs):
    train(
        unet,
        train_loader,
        optimizer=optimizer,
        loss_function=loss,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    current_val_loss = validate(
        unet, val_loader, loss, batchsize=5, step=step, tb_logger=logger, device=device
    )

    if current_val_loss < best_val_loss:
        # save the model
        best_val_loss = current_val_loss
        torch.save(unet.state_dict(), '/localscratch/DL4MIA_2024/BlastoSeg/saved_models/unet3d_model_stepsize2_best.pth')
