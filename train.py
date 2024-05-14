"""Definition of the training and validation loops"""

from dataset import (
    BlastoDataset
)

dataset = BlastoDataset("blastocyst_training")

print(dataset)
print(dataset.__getitem__(4))

