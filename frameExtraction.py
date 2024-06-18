import cv2
import os
import imutils
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import math
import torch

def extract():
    """
    For each .mp4 video in the input directory, tracks every moving object in frame, crops it, 
    and saves the image to a corresponding directory in the output directory.
    """

    torch.no_grad()

    video_paths = getVideoFilesNames()

    # Get the filepath of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    for video_file_name in video_paths:

        print("Working on " + video_file_name)

        os.chdir(dir_path) 

        video_file_path = dir_path + "/input/" + video_file_name
        images = analyzeVideo(video_file_path)

        # Create and switch to the output directory where we will store each frame,
        # which will be named ./output/<video_file_name>/frames/frame<counter>.png

        frames_directory = dir_path + "/output/" + video_file_name + "/frames"
        Path(frames_directory).mkdir(parents=True, exist_ok=True)
        os.chdir(frames_directory) 

        print("saving images...")

        counter = 0
        for image in images:
                # Save each extracted image
                cv2.imwrite(f'frame{counter}.png', image)
                counter += 1


def analyzeVideo(file_path):
    """
    Produces an array of images which contain moving objects from each frame of a video.
 
    Args:
        file_path (string): The file path of the video to analyze.
 
    Returns:
        array: An array of images which contains moving objects in each frame of the video.
    """

    # To return
    objects = []

    cap = cv2.VideoCapture(file_path)
    model =  YOLO("yolos/yolov8n.pt")

    while True:
    
        # Reading the content of the video by frames
        ret, frame = cap.read() 

        # Check if frame is not read correctly
        if not ret:
            break
        
        # Every frame goes through the YOLO model
        results = model(frame,stream=True) 

        for r in results: 
            
            # Creating bounding boxes
            boxes = r.boxes 
            
            for box in boxes:
                
                # Extracting coordinates
                x1, y1, x2, y2 = box.xyxy[0] 
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                
                # Creating instances width and height
                w,h = x2-x1,y2-y1 

                # Extract the image inside the bounding rectangle
                extracted_image = frame[y1:y1+h, x1:x1+w]

                objects.append(extracted_image)


    return objects



def getVideoFilesNames():
    """
    Gets the names of video files in the input directory.
 
    Returns:
        array: An array of strings representing the file names of .mp4 or .MOV video files in the input directory.
    """

    out = []

    # Get the filepath of the location of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Search for .mp4 files in /input
    for filename in os.listdir(dir_path + "/input"):
        if filename.endswith('.mp4') or filename.endswith('.MOV'):
            out.append(filename)

    return out



    
