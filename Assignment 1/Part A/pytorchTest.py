import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class NeuralNet(nn.Module):
    """
    neural network developed using torch package
    """

    def __init__(self, input_size, hidden_size, output_size):
        #initialize the neural network
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        #forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        #predict the output
        return self.forward(x)
    
    def loss(self, x, y):
        #calculate the loss
        return F.mse_loss(self.predict(x), y)

    def train(self, x, y, epochs, lr, batch_size, verbose = False):
        #train the model
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for i in range(0, x.size(0), batch_size):
                optimizer.zero_grad()
                output = self.loss(x[i:i+batch_size], y[i:i+batch_size])
                output.backward()
                optimizer.step()
            if verbose:
                print("Epoch: {}, Loss: {}".format(epoch, output.item()))
        return self
        
x = torch.randn(1000, 1)
y = 2*x +3 + torch.randn(1000, 1)

model = NeuralNet(1, 10, 1)

model.train(x,y,10,0.1,10, verbose=True)

out = model.predict(torch.tensor([[1.0]]))
#Expected output is tensor([[7.0]])

print(out)