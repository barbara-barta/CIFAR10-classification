import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import SixLayerPoolCNN
from dataset import get_loaders
from configs import class_names, model_name, batch_size
from utils import confusion_matrix_plot
from pathlib import Path

class CNN_CAM(SixLayerPoolCNN):
    def __init__(self):
        super().__init__()
        self.gradients=None

    def activations_hook(self,grad):
        self.gradients=grad

    def forward(self,x):
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x = torch.relu(x1+x2)
        x = self.pool(x)

        x3 = torch.relu(self.bn3(self.conv3(x)))
        x4 = self.bn4(self.conv4(x3))
        x = torch.relu(x3+x4)
        x = self.pool(x)

        x5 = torch.relu(self.bn5(self.conv5(x)))
        x6 = self.bn6(self.conv6(x5))
        x = torch.relu(x5+x6)

        x.register_hook(self.activations_hook)

        x = x.view(x.size(0),-1)
        return self.fc(x)


def gradcam(model,img,device):
    model.eval()
    out = model(img)
    cls = out.argmax(1)
    out[0,cls].backward()

    grads = model.gradients
    acts = model.conv5(img)

    pooled = torch.mean(grads,dim=[0,2,3])
    for i in range(acts.shape[1]):
        acts[:,i]*=pooled[i]

    heatmap = torch.mean(acts,dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= heatmap.max()
    return heatmap.detach().cpu().numpy()


def run_eval():
    device='cuda' if torch.cuda.is_available() else 'cpu'

    _,_,test_loader = get_loaders(batch_size,4)

    model = SixLayerPoolCNN().to(device)
    model_exists = list(Path(f"models/{model_name}").glob("*.tar"))
    model.load_state_dict(torch.load(model_exists[0],map_location=device))

    y_true,y_pred=[],[]

    for x,y in test_loader:
        x=x.to(device)
        out=model(x).argmax(1)
        y_true.extend(y.tolist())
        y_pred.extend(out.tolist())

    confusion_matrix_plot(y_true,y_pred,class_names)


if __name__=='__main__':
    run_eval()