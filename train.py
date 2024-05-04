#Open the UCF101 dataset, there is 20 examples for simplicity

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import UCF101
from torchvision.transforms import Lambda
from utils import parse_arguments, read_settings
from logger import Logger
from models import CNNRNN, ResNetGRU # First Two models are imported

from C3_model import C3DNetwork     #3D Convolutional Network model is imported

#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"


device  = torch.device('mps' if torch.backends.mps.is_available() else 'cuda') #Checking the model itsel by whether its working on local or on hyperion
print ('Device set to {0}'.format(device))



def custom_collate_fn(batch): # This custom code is taken from https://github.com/pytorch/vision/issues/2265
    filtered_batch = []       # The purpose of this code is to filter out the audio data from the batch
    for video, _, label in batch:
        filtered_batch.append((video, label))
    
    return torch.utils.data.dataloader.default_collate(filtered_batch)




def train(path_settings,train_settings):
    '''
    Training function that takes the path settings and training settings as input from the YAML file
    and trains a model on the UCF101 dataset

    Args:
    path_settings (dict): Dictionary containing the path settings
    train_settings (dict): Dictionary containing the training settings

    Returns:
     Models are saved in the models folder and parameters are saved in the checkpoints folder
    
    '''

    # Define a transform to preprocess the data
    transform = transforms.Compose([

            transforms.Lambda(lambda x: x / 255.),
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2) ), # reshape into (T, C, H, W) # T number of frames in the video clip , C is Channel, H is Height, W is Width
            transforms.Resize((240, 240)),      
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),     #This normalization used only for C3DNetwork model 
    ])

    
    # Load the UCF101 dataset
    print("ALl Data  loading may take a while ")
    train_dataset = UCF101(path_settings['root'], path_settings['annotation_path'], path_settings['frames_per_clip'], path_settings['step_between_clips'], fold=1, train=True,transform = transform )
    video, audio, label = train_dataset[0] #Get the first video and its label from training dataset

    print(f' Original Videos shape {video.shape}')    # (T, C, H, W)
    #print(f'Label: {label}')
    
    #print("ALL Data is batching now, may take a while ")
    train_loader = DataLoader(train_dataset, train_settings['batch_size'], num_workers = 0, shuffle=True, collate_fn=custom_collate_fn)     # Create a DataLoader
    #print("It is done, please check it ! ")

    #model = CNNRNN()
    #checkpoint_file = 'checkpoints/resnet50-19c8e357.pth'
    #model = ResNetGRU(10,checkpoint_file)
    model = C3DNetwork(10,path_settings['frames_per_clip']) #Constructor of the C3DNetwork model is called
                                                            #All model's constructions are different please take a look at the model.py file
                             
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.01) # Learning rate is changed to 0.01
    model.to(device)
   
    for epoch in range(train_settings['num_epochs']):
        model.train()


        for i, (video, label) in enumerate(train_loader):
   
            video = video.to(device)    
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(video)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')

        logger.log({'train_loss':loss.item()})
        print(f'Epoch {epoch}, Loss {loss.item()}')
    
    #Save the model parameters to a file
    torch.save({'epochs': epoch+1,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss.item()}, 'models/model_3DCN.pth')  #Path should be changed based on the model used
    
    print('Finished Training and saved the model')





if __name__ == '__main__':
    
    
    args = parse_arguments()
    settings = read_settings(args.config)  # Read the settings from the YAML file

    path_settings = settings['paths']
    train_settings = settings['training']
    print(path_settings['root'])           #Print the root and batch size for debugging purposes
    print(train_settings['batch_size'])

    #Logger experiment name changed manually for each different models 
    wandb_logger = Logger(f'Project_UCF101_3DCONV_b{train_settings["batch_size"]}_e{train_settings["num_epochs"]}', project='Action_Project')
    logger = wandb_logger.get_logger()

    train(path_settings, train_settings)




