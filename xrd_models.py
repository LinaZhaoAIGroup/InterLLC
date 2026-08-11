import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Encoder, EncoderLayer
from layers import FullAttention, AttentionLayer
from layers import DataEmbedding_inverted, PatchEmbedding


class XRDCNN1(nn.Module):
    def __init__(self, debug=False):
        super(XRDCNN1, self).__init__()      
        self.debug = debug

        self.cnn1 = nn.Conv1d(1, 80, kernel_size=100, stride=5, padding=48)
        self.dropout1 = nn.Dropout(0.3)
        self.avg_pool1 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.cnn2 = nn.Conv1d(80, 80, kernel_size=50, stride=5, padding=24)
        self.dropout2 = nn.Dropout(0.3)
        self.avg_pool2 = nn.AvgPool1d(kernel_size=3, stride=1)
        self.cnn3 = nn.Conv1d(80, 80, kernel_size=25, stride=2, padding=11)
        self.dropout3 = nn.Dropout(0.3)
        self.avg_pool3 = nn.AvgPool1d(kernel_size=3, stride=1)
 
        mlp_in_features = self._calculate_mlp_input_features()
        print(f"MLP输入特征维度: {mlp_in_features}")
        
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )
        
        self.shared_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    
    def _calculate_mlp_input_features(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 10001)
            x = F.relu(self.cnn1(x))
            if self.debug:
                print(f"After cnn1: {x.shape}")
            
            x = self.avg_pool1(x)
            if self.debug:
                print(f"After pool1: {x.shape}")

            x = F.relu(self.cnn2(x))
            if self.debug:
                print(f"After cnn2: {x.shape}")
            
            x = self.avg_pool2(x)
            if self.debug:
                print(f"After pool2: {x.shape}")

            x = F.relu(self.cnn3(x))
            if self.debug:
                print(f"After cnn3: {x.shape}")
            
            x = self.avg_pool3(x)
            if self.debug:
                print(f"After pool3: {x.shape}")
            
            features = x.numel()
            if self.debug:
                print(f"Total features before flatten: {features}")
            
            return features
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = F.interpolate(x, size=10001, mode='linear', align_corners=False)
        
        x = F.relu(self.cnn1(x))
        x = self.dropout1(x)
        x = self.avg_pool1(x)
        
        x = F.relu(self.cnn2(x))
        x = self.dropout2(x)
        x = self.avg_pool2(x)
        
        x = F.relu(self.cnn3(x))
        x = self.dropout3(x)
        x = self.avg_pool3(x)
        
        x = self.MLP(x)
        
        return x

class XRDCNN2(nn.Module):
    def __init__(self):
        super(XRDCNN2, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=50, stride=2, padding=25)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=25, stride=3, padding=12)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=3, padding=1)

        fc_input_size = 8064
        print(f"全连接层输入维度: {fc_input_size}")
        self.flatten = nn.Flatten()
        self.fcl1 = nn.Linear(fc_input_size, 2048)
        self.dropout1 = nn.Dropout(0.4)
        self.fcl2 = nn.Linear(2048, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fcl3 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.fcl4 = nn.Linear(128, 3) 

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        if x.shape[2] != 4501:
            x = F.interpolate(x, size=4501, mode='linear', align_corners=False)
        
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.leaky_relu(self.fcl1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fcl2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fcl3(x))
        x = self.dropout3(x)
        x = self.fcl4(x)  
        
        return x
    
class XRDCNN3(nn.Module):
    def __init__(self):
        super(XRDCNN3, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=20, stride=1, padding=9)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=10, stride=2, padding=5)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.fcl1 = nn.Linear(8064, 2500)
        self.fcl2 = nn.Linear(2500, 1000)
        self.fcl3 = nn.Linear(1000, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.fcl4 = nn.Linear(256, 3) 
        

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 
        
        x = F.interpolate(x, size=4501, mode='linear', align_corners=False)
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fcl1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fcl2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fcl3(x))
        x = self.dropout3(x)
        x = self.fcl4(x) 

        return x
    
class XRDCNN4(nn.Module):
    def __init__(self):
        super(XRDCNN4, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.flatten = nn.Flatten()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(8512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  
        )
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 
        
        x = F.interpolate(x, size=2251, mode='linear', align_corners=False)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        
        return x
    
class XRDCNN5(nn.Module):
    def __init__(self):
        super(XRDCNN5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2), 
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool1d(kernel_size=1),
            nn.Conv1d(8, 8, kernel_size=5, stride=2, padding=2),  
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool1d(kernel_size=1),
            nn.Conv1d(8, 4, kernel_size=5, stride=2, padding=2), 
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool1d(kernel_size=1)
        )
        self.dropout = nn.Dropout(0.4)
        fc_input_size = self._calculate_conv_output(10000)  
        print(f"卷积层输出维度: {fc_input_size}")
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  
        )
        
        self.input_size = 10000  


    def _calculate_conv_output(self, input_size):
        size = (input_size + 4 - 5) // 2 + 1 
        size = (size + 4 - 5) // 2 + 1  
        size = (size + 4 - 5) // 2 + 1  
        
        return 4 * size

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        if x.shape[2] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='linear', align_corners=False)
        
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    

class XRDCNN6(nn.Module):
    def __init__(self):
        super(XRDCNN6, self).__init__()
        self.input_size = 10000  
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=35, stride=1, padding=17)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=35, stride=1, padding=17)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=35, stride=1, padding=17)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=30, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=25, stride=1)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=25, stride=1)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=25, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc_input_size = self._calculate_fc_input()
        print(f"全连接层输入维度: {self.fc_input_size}")
        
        self.fcl1 = nn.Linear(self.fc_input_size, 500)
        self.fcl2 = nn.Linear(500, 250)
        
        self.fcl3 = nn.Linear(250, 128)
        self.fcl4 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(0.1)
        self.fcl5 = nn.Linear(32, 3)  
        

    def _calculate_fc_input(self):
        with torch.no_grad():
            x = torch.randn(1, 1, self.input_size)
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = self.pool1(F.leaky_relu(self.conv3(x)))
            x = self.pool1(F.leaky_relu(self.conv4(x)))
            x = self.pool1(F.leaky_relu(self.conv5(x)))
            x = self.pool1(F.leaky_relu(self.conv6(x)))
            x = self.pool1(F.leaky_relu(self.conv7(x)))
            x = self.flatten(x)
            return x.shape[1]

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 

        if x.shape[2] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='linear', align_corners=False)
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool1(F.leaky_relu(self.conv3(x)))
        x = self.pool1(F.leaky_relu(self.conv4(x)))
        x = self.pool1(F.leaky_relu(self.conv5(x)))
        x = self.pool1(F.leaky_relu(self.conv6(x)))
        x = self.pool1(F.leaky_relu(self.conv7(x)))
        x = self.flatten(x)
        x = self.fcl1(F.leaky_relu(x))
        x = self.fcl2(F.leaky_relu(x))
        x = self.dropout(x)
        
        x = F.leaky_relu(self.fcl3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fcl4(x))
        x = self.dropout(x)
        x = self.fcl5(x) 
        
        return x
    
class XRDCNN7(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(XRDCNN7, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=35, stride=1, padding=17),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=30, stride=1, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=25, stride=1, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=20, stride=1, padding=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=10, stride=1, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_input_size = self._calculate_fc_input()
        print(f"全连接层输入维度: {self.fc_input_size}")
        
        self.dense_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  
        )
        

    def _calculate_fc_input(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 4501)  
            x = self.conv_layers(x)
            x = self.flatten(x)
            return x.shape[1]

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = F.interpolate(x, size=4501, mode='linear', align_corners=False)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)

        return x
    
class XRDCNN8(nn.Module):
    def __init__(self, drop_rate=0.2, drop_rate_2=0.4):
        super(XRDCNN8, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.conv13 = nn.Conv1d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv1d(128, 3, kernel_size=1, stride=1)  
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate_2)
        self.apply(self.weight_init)
        
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
          
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = F.interpolate(x, size=8192, mode='linear', align_corners=False)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool(F.leaky_relu(self.conv5(x)))
        x = self.dropout(x)     
        x = self.pool(F.leaky_relu(self.conv6(x)))
        x = self.dropout(x)     
        x = self.pool(F.leaky_relu(self.conv7(x)))
        x = self.dropout(x)  
        x = self.pool(F.leaky_relu(self.conv8(x)))
        x = self.dropout(x)    
        x = self.pool(F.leaky_relu(self.conv9(x)))
        x = self.dropout(x)     
        x = self.pool(F.leaky_relu(self.conv10(x)))
        x = self.dropout(x)      
        x = self.pool(F.leaky_relu(self.conv11(x)))
        x = self.dropout(x)   
        x = self.pool(F.leaky_relu(self.conv12(x)))
        x = self.dropout2(x)   
        
        x = self.pool(self.conv13(x))
        x = self.dropout2(x)
        x = self.conv14(x)
        x = x.view(x.size(0), -1)
        
        return x
    
class XRDCNN9(nn.Module):
    def __init__(self):
        super(XRDCNN9, self).__init__()

        self.conv1 = nn.Conv1d(1, 27, kernel_size=11, stride=1)
        self.conv2 = nn.Conv1d(27, 27, kernel_size=11, stride=1)
        self.conv3 = nn.Conv1d(27, 27, kernel_size=11, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.21)
        self.dropout_fc1 = nn.Dropout(0.4)
        self.dropout_fc2 = nn.Dropout(0.3)
        self.dropout_fc3 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc_input_dim = None  
        self.fcl1 = None
        self.fcl2 = nn.Linear(2000, 512)
        self.fcl3 = nn.Linear(512, 128)
        self.fcl4 = nn.Linear(128, 3)  

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.dropout_conv(self.pool(F.relu(self.conv1(x))))
        x = self.dropout_conv(self.pool(F.relu(self.conv2(x))))
        x = self.dropout_conv(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        if self.fcl1 is None or self.fc_input_dim != x.shape[1]:
            self.fc_input_dim = x.shape[1]
            self.fcl1 = nn.Linear(self.fc_input_dim, 2000).to(x.device)
            print(f"初始化全连接层，输入维度: {self.fc_input_dim}")

        x = F.relu(self.fcl1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fcl2(x))
        x = self.dropout_fc2(x)
        x = F.relu(self.fcl3(x))
        x = self.dropout_fc3(x)
        x = self.fcl4(x) 
        
        return x
    

class XRDCNN10(nn.Module):
    def __init__(self, input_len=1000):
        super(XRDCNN10, self).__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=12, stride=1)
        self.conv2 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv3 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.conv4 = nn.Conv1d(24, 24, kernel_size=12, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.33)
        self.flatten = nn.Flatten()
        self.input_len = input_len
        self.fcl1 = None
        self.fcl2 = nn.Linear(2000, 3)  

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))
        x = self.flatten(x)

        if self.fcl1 is None:
            def calculate_output_size(length):
                length = (length - 12 + 1)  # conv1
                length = length // 2  # pool1
                length = (length - 12 + 1)  # conv2
                length = length // 2  # pool2
                length = (length - 12 + 1)  # conv3
                length = length // 2  # pool3
                length = (length - 12 + 1)  # conv4
                length = length // 2  # pool4
                return length
            
            actual_input_len = x.shape[2] if len(x.shape) == 3 else self.input_len
            conv_output_size = calculate_output_size(actual_input_len)
            fc_input_dim = conv_output_size * 24 
            
            print(f"实际输入序列长度: {actual_input_len}")
            print(f"计算出的全连接层输入维度: {fc_input_dim}")
            
            self.fcl1 = nn.Linear(fc_input_dim, 2000).to(x.device)
        
        if x.shape[1] != self.fcl1.in_features:
            def calculate_output_size(length):
                length = (length - 12 + 1)  # conv1
                length = length // 2  # pool1
                length = (length - 12 + 1)  # conv2
                length = length // 2  # pool2
                length = (length - 12 + 1)  # conv3
                length = length // 2  # pool3
                length = (length - 12 + 1)  # conv4
                length = length // 2  # pool4
                return length
            
            actual_conv_output = x.shape[1] // 24
            print(f"实际卷积输出维度: {actual_conv_output}")

            self.fcl1 = nn.Linear(x.shape[1], 2000).to(x.device)
            print(f"重新初始化全连接层，输入维度: {x.shape[1]}")

        x = F.relu(self.fcl1(x))
        x = self.dropout(x)
        x = self.fcl2(x)  
        
        return x
    

class XRDNoPoolCNN(nn.Module):
    def __init__(self):
        super(XRDNoPoolCNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(1, 80, 100, 5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(80, 80, 50, 5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(80, 80, 25, 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.CNN(x)

class XRDPredictor(nn.Module):
    def __init__(self, in_features, out_features=3):
        super(XRDPredictor, self).__init__()

        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 2300), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(2300, 1150), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(1150, out_features)  
        )

    def forward(self, x):
        return self.MLP(x)
    

class XRDCNN11(nn.Module):
    def __init__(self):
        super(XRDCNN11, self).__init__()
        self.cnn = XRDNoPoolCNN()
        
        mlp_in_features = 12160
        self.MLP = XRDPredictor(mlp_in_features, 3)  
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        target_size = 8500
        if x.shape[2] != target_size:
            min_size = 150 
            if x.shape[2] < min_size:
                x = F.interpolate(x, size=min_size, mode='linear', align_corners=False)

            x = F.interpolate(x, size=target_size, mode='linear', align_corners=False)
        
        x = self.cnn(x)
        x = self.MLP(x)
        return x


class XRDMLP(nn.Module):
    def __init__(self, input_size=3501):
        super(XRDMLP, self).__init__()
        self.input_size = input_size
        
        print(f"MLP模型输入维度: {self.input_size}")
        
        mlp_hidden1 = 7000 
        mlp_hidden2 = 7000
        if self.input_size < 50:
            mlp_hidden1 = 512
            mlp_hidden2 = 256
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.input_size, mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.flatten = nn.Flatten()
        fc_hidden = mlp_hidden2 // 2
        self.fc_layers = nn.Sequential(
            nn.Linear(mlp_hidden2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, 3),  
        )
    
        print(f"MLP模型结构:")
        print(f"  输入层: {self.input_size} -> {mlp_hidden1}")
        print(f"  隐藏层: {mlp_hidden1} -> {mlp_hidden2}")
        print(f"  输出层: {mlp_hidden2} -> {fc_hidden} -> 3")

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 3:
            if x.shape[1] == 1:  
                x = x.squeeze(1) 
            else:
                x = x.reshape(x.shape[0], -1) 
        elif len(x.shape) == 2:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        if x.shape[1] != self.input_size:
            current_size = x.shape[1]
            if abs(current_size - self.input_size) / self.input_size < 0.5:  
                print(f"调整输入维度: {current_size} -> {self.input_size}")
                x = x.unsqueeze(1)
                x = F.interpolate(
                    x, 
                    size=self.input_size, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1)  # [batch, seq_len]
            else:
                if current_size < self.input_size:
                    padding = self.input_size - current_size
                    x = F.pad(x, (0, padding), 'constant', 0)
                else:
                    x = x[:, :self.input_size]
        
        x = self.mlp_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
    

class XRDRNN(nn.Module):
    def __init__(self):
        super(XRDRNN, self).__init__()
        self.hidden_size = 64
        self.num_layers = 4
        
        self.rnn = nn.RNN(1, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 3)  

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1) 
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # [batch, hidden_size]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)  
        
        return out
    
class XRDLSTM(nn.Module):
    def __init__(self):
        super(XRDLSTM, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        
        self.lstm = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size, 3)  

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        x = x.unsqueeze(-1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.dropout(out)
        out = self.fc(out)  
        
        return out
    

class XRDGRU(nn.Module):
    def __init__(self):
        super(XRDGRU, self).__init__()
        self.hidden_size = 128 
        self.num_layers = 2    
        
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=0.2 
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size, 3)  

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # [batch, seq_len]
        
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out) 
        
        return out

class XRDBiRNN(nn.Module):
    def __init__(self):
        super(XRDBiRNN, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2 
        self.rnn = nn.RNN(
            input_size=1, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2  
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size * 2, 3) 

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  
        
        x = x.unsqueeze(-1) 
        out, _ = self.rnn(x)
        out = out[:, -1, :]  
        out = self.dropout(out)
        out = self.fc(out) 
        
        return out


class XRDBiLSTM(nn.Module):
    def __init__(self):
        super(XRDBiLSTM, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2  
        
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2  
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size * 2, 3)  

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # [batch, seq_len]
        
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out) 
        
        return out


class XRDBiGRU(nn.Module):
    def __init__(self):
        super(XRDBiGRU, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2 
        
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2  
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size * 2, 3)  

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # [batch, seq_len]
        
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)  
        
        return out
    

class XRDTransformer(nn.Module):
    def __init__(self):
        super(XRDTransformer, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        self.num_heads = 4
        
        self.embedding = nn.Linear(1, self.hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=self.num_heads,
            dim_feedforward=256,
            dropout=0.2,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)  

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)  
        
        return out
    

class XRDiTransformer(nn.Module):
    def __init__(self):
        super(XRDiTransformer, self).__init__()
        self.seq_len = 1000
        self.output_attention = False
        self.d_model = 512
        self.dropout = 0.1
        self.factor = 5
        self.n_heads = 2
        self.d_ff = 2048
        self.e_layers = 2
        self.activation = "gelu"
        self.enc_in = 1
        
        self.enc_embedding = DataEmbedding_inverted(
            c_in=self.seq_len,
            d_model=self.d_model,  
            embed_type='fixed', 
            freq='h',
            dropout=self.dropout
        )
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False, 
                            factor=self.factor, 
                            attention_dropout=self.dropout,
                            output_attention=self.output_attention
                        ), 
                        d_model=self.d_model, 
                        n_heads=self.n_heads
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    moving_avg=25,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            conv_layers=None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 3)  
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def regression(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
        
        x = F.interpolate(x, size=1000, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # [batch, 8192, 1]
        enc_out = self.enc_embedding(x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=None, delta=None)
        enc_out = enc_out[:, -1, :]  # [batch, d_model]
        output = self.fc_layers(enc_out)
        
        return output

    def forward(self, x):
        dec_out = self.regression(x)
        return dec_out  # [B, 3]


class XRDPatchTST(nn.Module):
    def __init__(self):
        super(XRDPatchTST, self).__init__()
        self.seq_len = 1000
        self.d_model = 256
        self.patch_len = 32 
        self.stride = 16 
        self.factor = 5
        self.dropout_rate = 0.1
        self.n_heads = 8
        self.d_ff = 512
        self.activation = 'gelu'
        self.e_layers = 2
        self.enc_in = 1
        self.output_attention = False
        
        padding = self.stride
        self.head_nf = self.d_model * int((self.seq_len - self.patch_len) / self.stride + 2)
        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, padding, self.dropout_rate)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, 
                            self.factor, 
                            attention_dropout=self.dropout_rate,
                            output_attention=self.output_attention
                        ), 
                        self.d_model, 
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout_rate,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.regression_head = nn.Sequential(
            nn.Linear(self.head_nf * self.enc_in, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(128, 3)  
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def regression(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 
        
        batch_size = x.shape[0]
        if x.shape[2] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        
        x = x.reshape(-1, self.seq_len, 1)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        x = x.permute(0, 2, 1)  # [batch, 1, seq_len]
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.regression_head(output)  # [batch, 3]
        
        return output

    def forward(self, x):
        return self.regression(x)

