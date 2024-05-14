"""Definition of the training and validation loops"""
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
import numpy as np
#device = torch.device("cuda") if torch.cuda.is_available() else 
device = torch.device("cpu")
assert torch.cuda.is_available()

from model import UNet

from dataset import (
    BlastoDataset
)

# Create training dataset
train_data = BlastoDataset("/group/dl4miacourse/projects/BlastoSeg/training")
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8)

val_data = BlastoDataset("/group/dl4miacourse/projects/BlastoSeg/validation")
val_loader = DataLoader(val_data, batch_size = 1, shuffle=True, num_workers=8)

unet = UNet(depth=4, in_channels=1, out_channels=4, num_fmaps=2).to(device)
optimizer = torch.optim.Adam(unet.parameters())


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
    batchsize = 5
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        #if torch.cuda.is_available():
        #    device = torch.device("cuda")
        #else:
        device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x_batch, y_batch) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # Loop over each slice in the batch
        for slice_id in range(0, x_batch.shape[1], batchsize):
            x = x_batch[:, slice_id:slice_id + batchsize, ...]
            y = y_batch[:, slice_id:slice_id + batchsize, ...]

            x = x.permute(1, 0, 2, 3)  # Assuming the first dimension is the batch dimension
            y = y.permute(1, 0, 2, 3)
            y = torch.squeeze(y,1)

            optimizer.zero_grad()

            # apply model and calculate loss
            prediction = model(x)  # Assuming model expects a batch dimension
            loss = loss_function(prediction, y)

            # backpropagate the loss and adjust the parameters
            loss.backward()
            optimizer.step()

            # log to console
            if batch_id % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_id * len(x_batch) + slice_id,  # Adjusted index calculation
                        len(loader.dataset),
                        100.0 * batch_id / len(loader),
                        loss.item(),
                    )
                )
            # log to tensorboard
            if tb_logger is not None:
                step = epoch * len(loader.dataset) + batch_id * len(x_batch) + slice_id  # Adjusted step calculation
                tb_logger.add_scalar(
                    tag="train_loss", scalar_value=loss.item(), global_step=step
                )
                # check if we log images in this iteration
                if step % log_image_interval == 0:
                    tb_logger.add_images(
                        tag="input", img_tensor=x.to("cpu"), global_step=step  # Assuming input requires batch dimension
                    )
                    tb_logger.add_images(
                        tag="target", img_tensor=y.to("cpu"), global_step=step  # Assuming target requires batch dimension
                    )
                    tb_logger.add_images(
                        tag="prediction",
                        img_tensor=prediction.to("cpu").detach(),
                        global_step=step,
                    )
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
        for x_batch, y_batch in loader:
            # move input and target to the activedevice
            for slice_id in range(0, x_batch.shape[1], batchsize):
                x = x_batch[:, slice_id:slice_id + batchsize, ...]
                y = y_batch[:, slice_id:slice_id + batchsize, ...]

                x = x.permute(1, 0, 2, 3)  # Assuming the first dimension is the batch dimension
                y = y.permute(1, 0, 2, 3)
                y = torch.squeeze(y,1)

                # apply model and calculate loss
                prediction = model(x)  # Assuming model expects a batch dimension
                val_loss += loss_function(prediction, y).item()

    # normalize loss and metric
    val_loss /= len(loader)

    print(
        "\nValidate: Average loss: {:.4f}".format(
            val_loss
        )
    )

loss = nn.CrossEntropyLoss(ignore_index=1)
n_epochs = 40
for epoch in range(n_epochs):
    train(
        unet,
        train_loader,
        optimizer=optimizer,
        loss_function=loss,
        epoch=epoch,
        log_interval=5,
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        unet, val_loader, loss,batchsize=5, step=step, device=device
    )

torch.save(unet.state_dict(), 'unet_model.pth')