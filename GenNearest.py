
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton


class GenNearest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """

    def __init__(self, videoSkeletonTarget):
        # Contains the skeleton of each frame from the target video
        self.videoSkeletonTarget = videoSkeletonTarget  

    def generate(self, skeleton):           
        """Generates an image from the given skeleton `skeleton`."""
        min_distance = float('inf')
        best_frame_idx = None

        # Iterate over each frame
        for frame_idx, video_skeleton in enumerate(self.videoSkeletonTarget.ske):
            # Calculate the distance
            distance = skeleton.distance(video_skeleton)

            # Update if the distance is smaller than the current minimum
            if distance < min_distance:
                min_distance = distance
                best_frame_idx = frame_idx

        # Return the image of the frame that minimizes the distance
        if best_frame_idx is not None:
            best_frame_image = self.videoSkeletonTarget.readImage(best_frame_idx)
            return best_frame_image
        else:
            print("Error: No matching frame found.")
            return np.zeros((128, 128, 3), dtype=np.uint8)  