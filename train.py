#Open the UCF101 dataset, there is 20 examples for simplicity

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



#Checking the model itsel by whether its working on local or on hyperion

#os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"

device  = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')

print ('Device set to {0}'.format(device))



def custom_collate_fn(batch): # This custom code is taken from https://github.com/pytorch/vision/issues/2265
    filtered_batch = []       # The purpose of this code is to filter out the audio data from the batch
    for video, _, label in batch:
        filtered_batch.append((video, label))
    
    return torch.utils.data.dataloader.default_collate(filtered_batch)



def train(path_settings,train_settings):

    #root = "UCF-101"
    #annotation_path = "ucfTrainTestlist"

    # Define a transform to preprocess the data
    transform = transforms.Compose([
             # scale in [0, 1]
             transforms.Lambda(lambda x: x / 255.),
             # reshape into (T, C, H, W) # T number of frames in the video clip , C is Channel, H is Height, W is Width

             transforms.Lambda(lambda x: x.permute(0, 3, 1, 2) ),       
    ])

    
    # Load the UCF101 dataset
    print("ALl Data  loading may take a while ")
    train_dataset = UCF101(path_settings['root'], path_settings['annotation_path'], frames_per_clip=5, step_between_clips=5, fold=1, train=True,transform = transform )
    #test_dataset = UCF101(root, annotation_path, frames_per_clip=5, step_between_clips=5, fold=1, train=False, transform = transform)






    
     #Get the first video and its label from training dataset
    video, audio, label = train_dataset[0]

    #print(video)
    print(f' Original Videos shape {video.shape}')    # (T, C, H, W)
    #print(f'Label: {label}')
    

    # Create a DataLoader
    print("ALL Data is batching now, may take a while ")
    train_loader = DataLoader(train_dataset, train_settings['batch_size'], num_workers = 0, shuffle=True, collate_fn=custom_collate_fn)
    print("It is done, please check it ! ")

    model = CNNRNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

    wandb_logger = Logger(f'INM705_Project_UCF101 {train_settings["batch_size"]}_lr{train_settings["num_epochs"]}', project='Action_Project')

    logger = wandb_logger.get_logger()

    model.to(device)
   
    for epoch in range(train_settings['num_epochs']):
        model.train()

        for i, (video, label) in enumerate(train_loader):
            video = video.to(device)    #.to(torch.float16) # Convert the video to float16 to solve the related mps issue
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
                'loss':loss.item()}, 'models/model_1.pth')
    
    print('Finished Training and saved the model')







    


if __name__ == '__main__':
    
    
    args = parse_arguments()

    # Read the settings from the YAML file
    settings = read_settings(args.config)

    path_settings = settings['paths']
    train_settings = settings['training']
    print(path_settings['root'])
    print(train_settings['batch_size'])

    train(path_settings, train_settings)




