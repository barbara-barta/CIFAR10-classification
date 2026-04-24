import torch
import matplotlib.pyplot as plt
from sklearn import metrics


def accuracy(pred, y):
    return (pred.argmax(1)==y).float().mean().item()


def test_model(model, loader, device):
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x).argmax(1)
            correct += (out==y).sum().item()
            total += y.size(0)
    return correct/total


def plot_curves(train_acc,val_acc):
    plt.plot(train_acc,label='train')
    plt.plot(val_acc,label='val')
    plt.legend()
    plt.show()


def confusion_matrix_plot(y_true,y_pred,class_names):
    cm = metrics.confusion_matrix(y_true,y_pred,normalize='true')
    disp = metrics.ConfusionMatrixDisplay(cm,display_labels=class_names)
    disp.plot()
    plt.show()