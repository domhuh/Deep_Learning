import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class BinaryBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.training_accuracy = []
        self.validation_accuracy = []

    def fit(self, x_train, y_train, x_test = None, y_test=None, epochs =2, lr=1e-3):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        pb = tqdm(range(epochs))
        for epoch in pb:
            self.train()
            optimizer.zero_grad()
            prediction = self(x_train)
            loss = criterion(prediction, y_train)
            self.losses.append(loss.item())
            loss.backward()
            optimizer.step()
            self.test(x_train,y_train,training=True)
            self.test(x_test,y_test,validation=True)
            pb.set_description(f"""{epoch} ||
                                   Loss: {round(self.losses[-1],2)} || 
                                   Accuracy: {round(self.training_accuracy[-1],2)} %
                                   """)
                
    def test(self, x, y, training = False, validation = False):
        self.eval()
        with torch.no_grad():
            prediction = self(x)
            correct = torch.sum(torch.round(prediction.squeeze()) == y)
            total = float(prediction.shape[0])
        if training:
            self.training_accuracy.append((correct/total * 100).item())
        if validation:
            self.validation_accuracy.append((correct/total * 100).item())
        return (correct/total * 100).item()