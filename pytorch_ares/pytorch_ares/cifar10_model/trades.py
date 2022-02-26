import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import torch
from third_party.trades.models.wideresnet import WideResNet
path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

model_path = os.path.join(path,'model/ciafar10_checkpoints/model_cifar_wrn.pt')

def load(device):
    model = WideResNet_TRADES(device)
    model.load()
    model.name = 'trades'
    return model


class WideResNet_TRADES(torch.nn.Module):
    def __init__(self, device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.model = WideResNet().to(self.device)

    def forward(self, x):
        output = self.model(x.to(self.device))
        return output

    def load(self):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.eval()


if __name__ == '__main__':
    
    
    if not os.path.exists(model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view'
        print('Please download "{}" to "{}".'.format(url, model_path))
