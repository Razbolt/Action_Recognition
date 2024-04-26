import torch
import torch.nn as nn

from torchvision.ops import RoIPool
from torchvision.datasets import UCF101

device  = torch.device('cpu' if torch.backends.mps.is_available() else 'cuda')
print('Device:',device)

#3C Convolutional Neural Network 

class C3DNetwork(nn.Module):
    def __init__(self, num_classes,in_channels, dropout_prob=0.5):
        super(C3DNetwork, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2,padding= 1)
        #Batch normalization
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2,padding=1)
        #Batch normalization
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2,padding=1)
        #Batch normalization
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        #Batch normalization
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0,1,1))
        #Batch normalization
        self.bn5 = nn.BatchNorm3d(512)

        # Fully connected layers
        self.fc6 = nn.Linear(512, 4096)
        self.fc7 = nn.Linear(4096, num_classes)  

        # Dropout layer
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(0.5)

        self.gap = nn.AdaptiveAvgPool3d(1) # GLOBAL average pooling

        #Constructors for Batch normalization before GRU and after GRU 

        #self.bn0 = nn.BatchNorm3d(512)
        self.relu = nn.ReLU()
        
        self.lstm = nn.GRU(input_size = 512, hidden_size = 128, num_layers = 4, batch_first = True)
        self.fc1 = nn.Linear(128, num_classes)

        # Apply weight initialization for better convergence 
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        # Pass through 3D convolutional layers
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))

        x = self.dropout1(x)  
        #print(f' After pool1 {x.shape}')

        
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)  
        #print(f' After pool2 {x.shape}')
        
        x = self.pool3(nn.functional.relu(self.conv3a(x)))
        x = self.dropout1(x)  
        #print(f' After pool3 {x.shape}')
        
        x = nn.functional.relu(self.conv3b(x))
        x = self.dropout2(x)  
        #print(f' After conv3b {x.shape}')
        
        x = self.pool4(nn.functional.relu(self.conv4a(x)))
        x = self.dropout2(x)  
        #print(f' After pool4 {x.shape}')
        
        x = nn.functional.relu(self.conv4b(x))
        x = self.dropout2(x)  
        #print(f' After conv4b {x.shape}')
        
        x = self.pool5(nn.functional.relu(self.conv5a(x)))
        x = self.dropout1(x)  
        #print(f' After pool5 {x.shape}')
        
        x = nn.functional.relu(self.conv5b(x))
        x = self.dropout1(x)  
        #print(f' After conv5b {x.shape}')
        

        x = self.gap(x)
        #print(f' After GAP {x.shape}')

        # Flatten the features
        x = x.view(x.size(0), -1)
        #print(f' After flattening {x.shape}')

        # Fully connected layers
        x = self.fc6(x)
        x = self.dropout3(x)
        x = self.fc7(x)
        #print(f' After FC7 {x.shape}')

        return x


def main():


    input = torch.randn(32, 16, 3, 240, 240).to(device)
    model = C3DNetwork(101,16)



    #print(model)
    model.to(device)
    model.train()
    output = model(input)

 


if __name__ == '__main__':
    main()
