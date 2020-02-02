from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RandomSamplingClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_accuracy, self.validation_accuracy = [0.0],[0.0]
        
    def fit(self, x_train, y_train, x_test=None, y_test=None,
            epochs = 1, lr = 1e-3, ne=5, verbose = True, split=None):
        criterion=nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        pb = tqdm(range(epochs)) if verbose else range(epochs)
        if not split is None:
            idx = int(x_train.shape[0]*split)
            x_test, y_test = x_train[idx:], y_train[idx:]
            x_train, y_train = x_train[:idx], y_train[:idx]
        for epoch in pb:
            for video, label in zip(x_train,y_train):
                self.train()
                optimizer.zero_grad()
                split_videos = list(torch.split(video,self.nf))                
                idxs  = np.random.choice(np.arange(len(split_videos)),ne)
                samples = torch.stack([split_videos[i] for i in idxs])
                pred = self(samples)
                label = label.expand(ne,1)
                loss = criterion(pred, label.unsqueeze(0))
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())
                if verbose: pb.set_description(f"""{epoch} ||
                                   Loss: {round(np.mean(self.losses[-5:]),2)} || 
                                   Accuracy: {round(np.mean(self.training_accuracy[-5:]) * 100,2)} %
                                   """)
            self.evaluate(x_train,y_train,training=True)
            if not x_test is None: self.evaluate(x_test,y_test,validation=True)

    
    def evaluate(self, x, y, training=False, validation=False, ne=5):
        self.eval()
        correct, total = 0.0, 0.0
        with torch.no_grad():
            for video, label in zip(x,y):
                split_videos = list(torch.split(video,self.nf))                
                idxs  = np.random.choice(np.arange(len(split_videos)),ne)
                samples = torch.stack([split_videos[i] for i in idxs])
                pred = self(samples)
                rounded = ((torch.tensor([0.5]).expand(ne,1).cuda() < pred)*1.0).squeeze()
                exp_label = label.expand(ne,1).squeeze()
                correct += sum(rounded.squeeze() == label.expand(ne,1).squeeze()).item()
                total+=ne
            if training: self.training_accuracy.append(correct/total)
            elif validation: self.validation_accuracy.append(correct/total)
        return correct/total