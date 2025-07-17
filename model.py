import torch
import torch.nn as nn

class Palm(nn.Module):
    def __init__(self):
        super(Palm, self).__init__()
        self.fc1 = nn.Linear(18,15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15,10)
        self.fc3 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class Index(nn.Module):
    def __init__(self):
        super(Index, self).__init__()
        self.fc1 = nn.Linear(12,15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15,10)
        self.fc3 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class Middle(nn.Module):
    def __init__(self):
        super(Middle, self).__init__()
        self.fc1 = nn.Linear(12,15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15,10)
        self.fc3 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
    
class Ring(nn.Module):
    def __init__(self):
        super(Ring, self).__init__()
        self.fc1 = nn.Linear(12,15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15,10)
        self.fc3 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class Pinky(nn.Module):
    def __init__(self):
        super(Pinky, self).__init__()
        self.fc1 = nn.Linear(12,15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15,10)
        self.fc3 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class Thumb(nn.Module):
    def __init__(self):
        super(Thumb, self).__init__()
        self.fc1 = nn.Linear(12,15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15,10)
        self.fc3 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
    
class Gesture(nn.Module):
    def __init__(self):
        super(Gesture, self).__init__()
        self.palm = Palm()
        self.thumb = Thumb()
        self.index = Index()
        self.middle = Middle()
        self.ring = Ring()
        self.pinky = Pinky()

        self.fc1 = nn.Linear(30,20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,10)

    def forward(self, x):
        opalm = self.palm(x[:,[0, 1, 2, 3, 4, 5, 15, 16, 17, 27, 28, 29, 39, 40, 41, 51, 52, 53]])
        othumb = self.thumb(x[:,3:15])
        oindex = self.index(x[:,15:27])
        omiddle = self.middle(x[:,27:39])
        oring = self.ring(x[:,39:51])
        opinky = self.pinky(x[:,51:63])

        result = torch.cat((opalm,
                            othumb,
                            oindex,
                            omiddle,
                            oring,
                            opinky), dim=1)
        
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(result)))))
