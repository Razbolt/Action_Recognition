import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models



device  = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')

#print ('Device set to {0}'.format(device))

class CNNRNN(nn.Module):  #Base Model for the project Vanilla CNNRNN model with 2 CNN layers and 3 LSTM layers
    def __init__(self):   #Constructor of the class only the layers are defined here in order to use for original data please change the last layer of self.fc2 to 101
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


        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 101) #You can change the last layer to 101 for the original data


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
        r_out = r_out[:, -1 , :]  #Use the output from the last time step
        r_out = self.dropout(r_out) #Apply dropout 

        x = self.fc1(r_out) #Pass data through fully connected layers
        output= self.fc2(x)

        return output
    

class ResNetGRU(nn.Module):

    def __init__(self, num_classes,checkpoint_file=None): #Constructor of the class , num_classes is the number of classes in the dataset
        super(ResNetGRU, self).__init__() #Call the constructor of the parent class
        
        
       
        if checkpoint_file is not None:  #Load the pretained model if the checkpoint file is provided
            state_dict = torch.load(checkpoint_file)
            print('Model loaded from checkpoint,hopefully :)')
        else:
            state_dict = models.resnet50(pretrained=False).state_dict() #Change it to True whenever you are training for the first time

      
        self.model  = models.resnet50()   #Load the resnet50 model
        self.model.load_state_dict(state_dict) #Load the weights into the model
        
        
        self.encoder = torch.nn.Sequential(*list(self.model.children())[:-1]) # Removing the last layer 
                                                                              # Also contains all layers except the last one

        
         
        self.bottleneck = nn.Linear(2048 , 2048) #Include bottle neck layer between encoder and gru
        self.bn = nn.BatchNorm1d(2048) # Batch normalization 
        self.relu = nn.ReLU()
        self.gru = nn.GRU(2048, 512, num_layers=3, batch_first=True, dropout=0.5)

        self.bn2 = nn.BatchNorm1d(512)
        self.fc0 = nn.Linear(512,num_classes)
        
        #Print the last layer of encoder 
        #print(self.encoder[0])
        #print(self.classifier)


    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.float()
        x = x.reshape(batch_size * timesteps, C, H, W)
        
        x = self.encoder(x)
        #print(f' After encoder shape is: {x.shape}')

        x = x.view(x.size(0), -1) # Flatten the output look it tomorrow detailed
        x= self.bottleneck(x)
        x= self.bn(x)             # First batch normalization applied
        #print(f' After bottleneck shape is: {x.shape}')
        x = self.relu(x)
        #print(f' After ReLU shape is: {x.shape}')

        x = x.reshape(batch_size, timesteps, -1) # Modifing the input in order to pass it to gru 
                                                 # GRU expects 3D tensor as  ( batch_size,sequence_length and num_feautres)
        gru_out, _ = self.gru(x)

        #Take the output from the last time step        # CHECK IT TOO!!!
        gru_out = gru_out[:, -1, :]
        gru_out = self.bn2(gru_out) # Second batch normalization applied before fully connected layer

        x = self.fc0(gru_out)

        return x

        
        




        
        


if __name__ == '__main__': #Testing the models

    input = torch.rand(32,5,3,240,320).to(device)

    checkpoint_file = 'checkpoints/resnet50-19c8e357.pth'
    resnetgru = ResNetGRU(10,checkpoint_file)
            
    resnetgru.to(device)
    resnetgru.train()
    output = resnetgru(input)

    print('Model uploaded')
    
    




