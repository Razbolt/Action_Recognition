import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.checkpoint import checkpoint

# Create a first simple CNN and RNN model to train on the UCF101 dataset
# TODO:
# Also I can resize the pixels to 128x128 or 64x64
# Backbone from other models' parameters can be frozen and used as a feature extractor
# Transformers for decode parth 
# 3D convolutional neural network 

#Faster RNN ROI and attention ??? RPN

class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN,self).__init__()


        #CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc0 = nn.Linear(307200,64)
        self.lstm = nn.LSTM(input_size = 64, hidden_size = 128, num_layers = 3, batch_first = True)
        self.dropout = nn.Dropout(0.3)

        #For simplicity I only have 10 classes
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 101)


    def forward(self,x):
        batch_size , timespets , C,H,W = x.size()
        x = x.reshape(batch_size * timespets,C,H,W) # Convert 5D tensor to 4D tensor
        #print(f' Before CNN {x.shape}')
    

        #Pass data through CNN
        c_out = self.cnn(x)
        #print(f' After CNN {c_out.shape}')

        #Reshape the output to (batch_size, timespets, features) -1 mens channels * height * width
        
        r_in = c_out.reshape(batch_size,timespets,-1)  # Convert 4D tensor to 3D tensor
        #print(f' Getting read to pass data through LSTM {r_in.shape}')

        r_in = self.fc0(r_in)
        #print(f' Before really getting to LSTM  {r_in.shape}')
        #Pass data through LSTM
        r_out, (h_n, c_n) = self.lstm(r_in) # 3D tensor of output (batch_size, timespets, features)

        #Use the output from the last time step
        r_out = r_out[:, -1 , :]

        #Apply dropout 
        r_out = self.dropout(r_out)

        #Pass data through fully connected layers
        x = self.fc1(r_out)
        output= self.fc2(x)

        return output







