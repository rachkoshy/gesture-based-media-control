import numpy as np 
import pickle
from model import Gesture
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

data_labels_dict = pickle.load(open('data.pickle', 'rb'))

data = data_labels_dict["data"]
labels = data_labels_dict["labels"]


data = np.array(data)

print(data.shape)
labels = np.array(labels)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


num_epochs = 150
learning_rate = 0.01
batch_size = 1000
model = Gesture()
# model.load_state_dict(torch.load('Gesture_model3d.pth'))

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

custom_dataset = CustomDataset(data, labels)
data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
n_total_steps = len(custom_dataset)

# loss_vals=[]

for epoch in range(num_epochs):
    epoch_loss= []
    for i, (data, labels) in enumerate(data_loader):
        # print("data is of shape: ", data.shape)
        outputs = model(data)

        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)

        #losses.append(loss)
        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%1 == 0:
            print(f"epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item():.6f}")
    # loss_vals.append(sum(epoch_loss)/len(epoch_loss))

torch.save(model.state_dict(), 'Gesture_new.pth')

# plt.plot(range(num_epochs), loss_vals)
# plt.show()
