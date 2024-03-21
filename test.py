#Test the pytorch model from the models folder
import torch 

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import UCF101
from torchvision.transforms import Lambda
from utils import parse_arguments, read_settings
from logger import Logger

from models import CNNRNN


#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"

device  = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')

print ('Device set to {0}'.format(device))




def custom_collate_fn(batch): # This custom code is taken from https://github.com/pytorch/vision/issues/2265
    filtered_batch = []       # The purpose of this code is to filter out the audio data from the batch
    for video, _, label in batch:
        filtered_batch.append((video, label))
    
    return torch.utils.data.dataloader.default_collate(filtered_batch)



# Define a transform to preprocess the data
transform = transforms.Compose([
    # scale in [0, 1]
    transforms.Lambda(lambda x: x / 255.),

    # reshape into (T, C, H, W) # T number of frames in the video clip , C is Channel, H is Height, W is Width
    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2) ),

    ])

def test(path_settings):
   
    
    test_dataset = UCF101(path_settings['root'], path_settings['annotation_path'], frames_per_clip=5, step_between_clips=5, fold=1, train=False, transform = transform)
    test_loader = DataLoader(test_dataset, batch_size=32,num_workers=0, shuffle=True,collate_fn=custom_collate_fn)

    #Change it whenever you want to test another model this is for our base model
    checkpoint = torch.load('models/model_1.pth')

    model = CNNRNN()
    model.to(device)

    #
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Testing the model may take a while')
    model.eval() # Setting the model to evaluation mode 

    correct = 0
    total = 0 

   
    with torch.no_grad(): # Disable the gradient calculation
    
        for inputs,labels in test_loader:
            inputs= inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            #print(outputs)

            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test: {accuracy} %')
            #Get the predicted class



if __name__ == '__main__':


    args = parse_arguments()

    #Read the settings from the YAML file
    settings = read_settings(args.config)

    path_settings = settings['paths']

    print(path_settings['root'])

    test(path_settings)
        


