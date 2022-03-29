import os
import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from dataset import DataHandler
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, CenterCrop, RandomCrop
import torch
import numpy as np



class MNISTData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args

    def train_dataloader(self):
        transform = ToTensor()
        dataset = MNIST("mnist", train=True, download=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = ToTensor()
        dataset = MNIST("mnist", train=False, download=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def save_train_data(self, trainloader, path):
        # TODO follow this, modify transform(contain only totensor) to get sprite image, the order should be preserved
        trainset_data = None
        trainset_label = None
        all_idxs = None

        for batch_idx, (inputs,targets) in enumerate(trainloader):
            if trainset_data != None:
                # print(input_list.shape, inputs.shape)
                trainset_data = torch.cat((trainset_data, inputs), 0)
                trainset_label = torch.cat((trainset_label, targets), 0)
                # all_idxs = torch.cat((all_idxs, idxs), 0)
            else:
                trainset_data = inputs
                trainset_label = targets
                # all_idxs = idxs
        
        training_path = os.path.join(path, "Training_data")
        if not os.path.exists(training_path):
            os.mkdir(training_path)
        torch.save(trainset_data, os.path.join(training_path, "training_dataset_data.pth"))
        torch.save(trainset_label, os.path.join(training_path, "training_dataset_label.pth"))
        torch.save(all_idxs, os.path.join(training_path, "training_dataset_idxs.pth"))


    def save_test_data(self, testloader, path):

        testset_data = None
        testset_label = None
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if testset_data != None:
                # print(input_list.shape, inputs.shape)
                testset_data = torch.cat((testset_data, inputs), 0)
                testset_label = torch.cat((testset_label, targets), 0)
            else:
                testset_data = inputs
                testset_label = targets

        testing_path = os.path.join(path, "Testing_data")
        if not os.path.exists(testing_path):
            os.mkdir(testing_path)
        torch.save(testset_data, os.path.join(testing_path, "testing_dataset_data.pth"))
        torch.save(testset_label, os.path.join(testing_path, "testing_dataset_label.pth"))
        idxs = [i for i in range(len(testset_label))]
        torch.save(torch.Tensor(idxs), os.path.join(testing_path, "testing_dataset_idxs.pth"))

