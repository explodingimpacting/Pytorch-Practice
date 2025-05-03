"""
Description:
Pytorch practice building a neural network using mnist dataset.
"""
import torch # type: ignore
from torch import nn #type: ignore
from torch.utils.data import DataLoader #type:ignore
from torchvision import datasets #type:ignore
from torchvision.transforms import ToTensor #type:ignore

#x = torch.rand(5, 3)
#print(x)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


#download training data from open datasets
training_data = datasets.FashionMNIST(
    root = "data", 
    train = True, 
    download = True, 
    transform = ToTensor(),
)

#download test data from open datasets.
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True, 
    transform = ToTensor(),
)

batch_size = 64

#create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N (batch size), C (channels), H(height), W(width)]: {X.shape}")
    print(f"Shape of y (int labels for each image): {y.shape} {y.dtype} \n")
    print(f"a 4 D tensor of: batch size -  {batch_size} total images \n {X.shape[1]} amount of channels (if 1 then grayscale) \n {X.shape[2]} & {X.shape[3]} image height and width \n")
    break

#define a neural network; define layers of net and specify how data will pass through
# the net in the forward function
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #remember flattening turns into 1D array
        #3 layer network with 512 neurons each
        #ReLU activations; 10 output nodes
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # input layer to hidden
            nn.ReLU(),
            nn.Linear(512, 512), # hidden layer to hidden
            nn.ReLU(),
            nn.Linear(512, 10) # hidden to output with 10 outputs (0-9 digits)
        )

#describes how data moves through the model (logits are raw scores before softmax)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

#Optimizing model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #computer predicition error
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# we also check models performance against the test dataset to ensure it is learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("DONE!!")


#saving model
#common way to save to serialize the internal state dictionary (containing model parameters)
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#to load model
"""
The process for loading a model includes re-creating the model structure and loading
the state dictionary into it."""

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))


#This model can now be used to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
#to change image to test on simply change the index :)
#test_data[index][0], test_data[index][1]
x, y = test_data[9][0], test_data[9][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
