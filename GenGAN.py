
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 

class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Output a single value per input
            nn.Flatten(),  # Flatten to (batch_size, 1)
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            checkpoint = torch.load(self.filename)
            self.netG.load_state_dict(checkpoint['netG'])  # Carga solo los pesos del generador
            self.netD.load_state_dict(checkpoint['netD'])  # Carga los pesos del discriminador si es necesario


    def train(self, n_epochs):
        criterion = nn.BCELoss()
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        print("Starting Training")
        print(n_epochs)
        for epoch in range(n_epochs):
            print(epoch)
            for i, (skeletons, real_images) in enumerate(self.dataloader):
                real_images = real_images
                skeletons = skeletons

                # Train Discriminator
                self.netD.zero_grad()
                label = torch.full((real_images.size(0),), self.real_label, dtype=torch.float)
                output = self.netD(real_images).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()

                noise = skeletons
                fake_images = self.netG(noise)
                label.fill_(self.fake_label)
                output = self.netD(fake_images.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                optimizerD.step()

                # Train Generator
                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake_images).view(-1)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()

            # Save model checkpoint after each epoch
            torch.save({'netG': self.netG.state_dict(), 'netD': self.netD.state_dict()}, self.filename)
            print(f"Epoch {epoch+1} complete. Model saved.")




    def generate(self, ske):           # TP-TODO
        """ generator of image from skeleton """
        ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten())
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.view(1, Skeleton.reduced_dim, 1, 1)  # Adjust to match GenNNSkeToImage input shape
        self.netG.eval()
        with torch.no_grad():
            normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(500) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

