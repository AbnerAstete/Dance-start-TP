import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Uncomment for TensorBoard logging
#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)

class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        # Create a white image of size (image_size x image_size)
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        # Draw skeleton onto the image
        ske.draw(image)
        # Convert color format from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """
        VideoSkeleton dataset:
        - videoSke: VideoSkeleton object containing video and skeleton data for each frame
        - ske_reduced: Boolean to decide if reduced skeleton (13 joints x 2 dimensions) or full skeleton (33 joints x 3 dimensions) is used
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ", "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ", Skeleton.full_dim, ")")

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        # Preprocess skeleton (input)
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        
        # Preprocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            # Convert skeleton to tensor and flatten it
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            # Reshape tensor for neural network input
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        # Convert tensor back to a numpy image
        numpy_image = normalized_image.detach().numpy()
        # Rearrange dimensions from (C, H, W) to (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # Convert to OpenCV format (BGR) for display
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        # Denormalize image
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        return denormalized_image

def init_weights(m):
    # Initialize weights based on layer type
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GenNNSkeToImage(nn.Module):
    """ 
    Class that generates a new image from a video skeleton (skeleton posture).
    Function generator(Skeleton) -> Image
    """
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()
        )
        self.model.apply(init_weights)  
        print(self.model)

    def forward(self, z):
        # Reshape the input tensor to match the expected input dimensions
        z = z.view(z.size(0), 26)
        img = self.model(z)
        # Reshape the output to image dimensions (batch_size, 3, 64, 64)
        img = img.view(-1, 3, 64, 64)
        return img

class GenVanillaNN():
    """ 
    Class that generates a new image from video skeleton (skeleton posture).
    Function generator(Skeleton) -> Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.netG = GenNNSkeToImage()  # Initialize the generator network
        self.filename = 'data/Dance/DanceGenVanillaFromSke.pth'

        # Transformations for the target image (resize, crop, normalize)
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # Initialize the dataset with video skeleton and transformations
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        
        # Load pretrained model if available
        if loadFromFile and os.path.isfile(self.filename):
            self.netG.load_state_dict(torch.load(self.filename))

    def train(self, n_epochs):
        # Set the model to training mode
        self.netG.train()

        # Set up optimizer and loss function
        self.optimizer = optim.Adam(self.netG.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            print(epoch)
            running_loss = 0.0

            # Training loop
            for i, (inputs, labels) in enumerate(self.dataloader, 0):
                inputs, labels = inputs, labels
                self.optimizer.zero_grad()  # Zero the gradients

                # Forward pass through the model
                outputs = self.netG(inputs)

                # Calculate the loss
                loss = self.criterion(outputs, labels)
                loss.backward()  # Backpropagate the loss

                # Update model parameters
                self.optimizer.step()
                running_loss += loss.item()

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            torch.save(self.netG.state_dict(), self.filename)  

        print('Finished Training')

    def generate(self, ske):
        """ 
        Generator to produce image from skeleton
        """
        self.netG.eval()  
        ske_t = self.dataset.preprocessSkeleton(ske)  # Preprocess skeleton
        ske_t_batch = ske_t.unsqueeze(0)  # Add batch dimension
        normalized_output = self.netG(ske_t_batch)  # Generate image
        res = self.dataset.tensor2image(normalized_output[0])  # Convert tensor to image
        return res


if __name__ == '__main__':
    force = False
    optSkeOrImage = 2           # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 2000  # 200
    #train = 1 #False
    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
