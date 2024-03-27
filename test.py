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

from models import CNNRNN, ResNetGRU

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 


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

    transforms.Resize((240, 320))

    ])

def test(path_settings,train_settings):
   
    
    test_dataset = UCF101(path_settings['root'], path_settings['annotation_path'], path_settings['frames_per_clip'], path_settings['step_between_clips'], fold=1, train=False, transform = transform)
    test_loader = DataLoader(test_dataset, train_settings['batch_size'],num_workers=0, shuffle=True,collate_fn=custom_collate_fn)


    video, audio, label = test_dataset[0]

    #print(video)
    print(f' Original Videos shape {video.shape}')    # (T, C, H, W)
    #print(f'Label: {label}')

    #Change it whenever you want to test another model this is for our base model
    checkpoint = torch.load('models/model_1.pth',map_location=torch.device('cpu'))

    #The model tested in hyperion !! Check hyperion file

    model = CNNRNN()
    model.to(device)

    #
    model.load_state_dict(checkpoint['model_state_dict'])

    
    model.eval() # Setting the model to evaluation mode 

    correct = 0
    total = 0 

    
    failed_tests = 0

    y_true = []
    y_pred = []

    with torch.no_grad(): # Disable the gradient calculation
        try:
            for inputs,labels in test_loader:
                print('Testing the model may take a while')
                inputs= inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                #print(outputs)

                _, predicted = torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        except RuntimeError as e:
            failed_tests += 1
            print(f'Failed to test the model {failed_tests} times')
            print(f'Error: {e}')
            pass
    
    #Calculating the confusion matrix and classification report
    cm = confusion_matrix(y_true,y_pred)
    report = classification_report(y_true,y_pred,zero_division=1)

    with open('test_results/classification_report.txt','w') as f:
        f.write(report)
    
    #Plot the confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix',size = 15)
    plt.savefig('test_results/confusion_matrix.png')

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test: {accuracy} %')



if __name__ == '__main__':


    args = parse_arguments()

    #Read the settings from the YAML file
    settings = read_settings(args.config)

    path_settings = settings['paths']
    train_settings = settings['training']

    print(path_settings['root'])
    print(train_settings)

    test(path_settings,train_settings)
        


