import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import torch
from third_party.trades.models.wideresnet import WideResNet
path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

model_path = os.path.join(path,'model/ciafar10_checkpoints/model_best.pth')


def load(device):
    model = Label_Smoothing(device)
    model.load()
    model.name = 'label_smoothing'
    return model

class Label_Smoothing(torch.nn.Module):
    def __init__(self,device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.model = WideResNet().to(device)
        self.model = torch.nn.DataParallel(self.model)
        self._mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(device)
        self._std_torch = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).to(device)

    def forward(self, x):
        input_var = (x.to(self.device) - self._mean_torch) / self._std_torch
        labels = self.model(input_var)
        return labels

    def load(self):
        checkpoint = torch.load(model_path)
        # print(checkpoint["test_robust_acc"], checkpoint["test_acc"])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.eval()

if __name__ == '__main__':
    if not os.path.exists(model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://drive.google.com/file/d/1L2uf0eyScf7rVYZ1vEXLaszjpkhFlB63/view'
        print('Please download "{}" to "{}".'.format(url, model_path))