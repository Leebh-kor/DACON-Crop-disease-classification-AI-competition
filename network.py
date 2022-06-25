import torch
import torch.nn as nn
import timm

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.encoder_name, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 7)
    
    def forward(self, x):
        x = self.encoder(x)
        return x


class Network_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True,
                                    drop_path_rate=0,
                                    )
        
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 7)

    def forward(self, x):
        x = self.encoder(x)
        return x
