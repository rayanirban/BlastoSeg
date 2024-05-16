"""Definition of the training and validation loops"""
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
weights = [0.5, 1.0, 2.5, 6.0]
class_weights = torch.FloatTensor(weights).cuda() 
optimizer = torch.optim.Adam(unet.parameters(),lr=0.001)


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    device=None,
    early_stop=False,
    batchsize = 5
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
    av_loss = 0
    # Iterate over the batches of this epoch
    for batch_id, (x_batch, y_batch) in enumerate(loader):
        loss= 0
        # Move input and target to the active device (either cpu or gpu)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Loop over each slice in the batch
        for slice_id in range(0, x_batch.shape[1], batchsize):
            x = x_batch[:, slice_id:slice_id + batchsize, ...]
            y = y_batch[:, slice_id:slice_id + batchsize, ...]

            x = x.permute(1, 0, 2, 3)  # Assuming the first dimension is the batch dimension
            y = y.permute(1, 0, 2, 3)
            y = torch.squeeze(y, 1)

            optimizer.zero_grad()

            # Apply model and calculate loss
            prediction = model(x)  # Assuming model expects a batch dimension
            loss += loss_function(prediction, y)
        
        # Compute average loss for this batch and add to total loss
        loss /= x_batch.shape[1]
        
        # Backpropagate the total loss and adjust the parameters
        loss.backward()
        optimizer.step()
        av_loss+=loss
    av_loss/= len(loader)
    return x, y, prediction, av_loss
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
    av_loss = 0
    # Disable gradients during validation
    with torch.no_grad():
        # Iterate over validation loader and update loss values
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Initialize temporary loss accumulator for this batch
            val_loss = 0
            
            # Loop over each slice in the batch
            for slice_id in range(0, x_batch.shape[1], batchsize):
                x = x_batch[:, slice_id:slice_id + batchsize, ...]
                y = y_batch[:, slice_id:slice_id + batchsize, ...]

                x = x.permute(1, 0, 2, 3)  # Assuming the first dimension is the batch dimension
                y = y.permute(1, 0, 2, 3)
                y = torch.squeeze(y, 1)

                # Apply model and calculate loss
                prediction = model(x)  # Assuming model expects a batch dimension
                val_loss += loss_function(prediction, y)
            # Compute average loss for this batch and add to validation loss
            val_loss /= x_batch.shape[1]
            av_loss+=val_loss
        av_loss/= len(loader)
    return x, y, prediction, av_loss

logger = SummaryWriter("runs/Unet")
loss = nn.CrossEntropyLoss(ignore_index=1, weight=class_weights)
n_epochs = 3000
best_val_loss = 1000
best_epoch = -1

for epoch in range(n_epochs):
    trainx, trainy, trainprediction, trainloss= train(
        unet,
        train_loader,
        optimizer=optimizer,
        loss_function=loss,
        epoch=epoch,
        batchsize=5,
        log_interval=5,
        device=device,
    )
                    # log to console
    print(
        "Train Epoch: {} Training Loss: {:.6f}".format(
            epoch,  # Adjusted index calculation
            trainloss,
        ))
    val_x, val_y, val_prediction, val_loss = validate(
        unet, val_loader, loss, batchsize=5, device=device
    )
    print(
        "\nValidate: Average loss: {:.4f}".format(
            val_loss
        )
    )
    if logger is not None:
        trainy=trainy.unsqueeze(1)
        trainprediction = torch.sum(trainprediction, dim=1).unsqueeze(1)

        

        logger.add_scalars('Run', {"train_loss": trainloss, "val_loss": val_loss}, epoch)

        # check if we log images in this iteration

        logger.add_images(
            tag="input", img_tensor=trainx.cpu().numpy(), global_step=epoch
        )
        logger.add_images(
            tag="target", img_tensor=trainy.cpu().numpy(), global_step=epoch
        )
        logger.add_images(
            tag="prediction",
            img_tensor=trainprediction.to("cpu").detach(),
            global_step=epoch,
        )

    # Save the model if validation loss is improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(unet.state_dict(), '/group/dl4miacourse/projects/BlastoSeg/boundary_prediction_lr00001_weighted.pth')

print(f"Best model saved with validation loss: {best_val_loss} at epoch {best_epoch}")
