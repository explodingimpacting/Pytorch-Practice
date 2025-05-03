Any relevant notes in relation to Nueral Network implementation: 

- batch size is amount of images in said batch
- flatten() turn insto 1d array
- defining: 
self.linear_relu_stack = nn.Sequential(
    nn.Linear(28*28, 512),  # input layer → hidden
    nn.ReLU(),
    nn.Linear(512, 512),    # hidden → hidden
    nn.ReLU(),
    nn.Linear(512, 10)      # hidden → output
)
- we are inputing an image of 28x28 pixels and defining 512 nodes or neurons each
- With 10 outputs for digits (0-9)

With this:

def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

We are defining how data moves through the model. Logits are raw scores before softmax is called. 

Lets clarify softmax:
- softmax turns raw outputs scores (logits) into probabilities.
- Ensures: all output vales are between 0 and 1, all values sum to 1; the highest value corresponds to most likely class

ex: 

logits = [2.0, 1.0, 0.1]

After we apply softmax, you get:

probabilities = [0.65, 0.24, 0.11] 

This means that class 0 has a probability of 65%, class 1: 24% and class 2: 11%.
These can be interpreted as confidence scores for each class.

In pytorch, using nn.CrossEntropyLoss() applies softmax inside, it expects raw logits not probabilities.

When do we want to see probabilitites?
- during evaluation
- during visualization
- can apply softmax like this:
    probs = torch.softmax(logits, dim=1)
    - dim = 1 means your applying softmax across class (not across batch)

When calling nn.Linear(image size, num of nuerons):
- There is no strict rule for choosing this number (this is a hyperparameter)
- 512 is commonly used. 
    - Power of 2: Computers are optimized for powers of 2 (128, 256, 512, 1024..). Can be efficient on hardware.
    - large enough to learn complex patterns from 784 input pixels.
    - small enough to avoid huge memory use or overfitting from small datasets like MNIST.
    - Emperically effective
    - nn.Linear(image size, 256) has fewer neurons and will be FASTER but might UNDERFIT.
    - nn.Linear(Image size, 1024) has more neurons and may overfit, be slower.

Optmizing Model Parameters:
- To train a model, we need a loss function and an optimizer.
- In a single training loop, model makes predicitions on training dataset (fed to it in batches) and backpropogates the predicition error to adjust the models parameters.


Training: 
- a single training loop, model makes predicitions on training dataset (fed to it in batches), and backpropogtes the prediction error to adjust the models parameters.

- we also check the models performance against the test dataset to ensure it is learning

- Training process is conducted over several iterations (epochs). During each epoch, the model learns parameters to make better predictions. We print the models accuracy and loss at each epoch; wed like to see the accuracy increase and loss decrease with every epoch. 

- Training involves:
    - Forward pass: Model make predictions on the input batch.
    - Loss calculation: How far off are the predictions from the true labesl?
    - Backward pass: The model computes gradients (how much each weight contributed to the loss)
    - Optimization step: Adjusts the weights to reduce loss
    - All of this is done batch by batch over entire dataset.

- Epoch: One full pass through the entire training dataset (ex: 60,000 images)
- Batch: A small group of samples passed at once (ex: batchsize = 64)
- Step: one update from one batch
- So during one epoch, you'll have 60,000/ 64 = 938 steps.
- ex print output: 

    Epoch 3
    -------------------
    loss: 1.923680 [   64/60000]
    loss: 1.890846 [ 6464/60000]
    loss: 1.773679 [12864/60000]
    loss: 1.801109 [19264/60000]
    loss: 1.724114 [25664/60000]
    loss: 1.659627 [32064/60000]
    loss: 1.675920 [38464/60000]
    loss: 1.579575 [44864/60000]
    loss: 1.599333 [51264/60000]
    loss: 1.488184 [57664/60000]
    Test error: 
    Accuracy: 61.1%, Avg loss: 1.512071 

    We choose to print every 100~ steps.
    6464/64 = 101 
    12664/64 = 201
    19264/64 = 301
    and so on..
    In this printout: 
    - the current loss on batch being processed
    - model has seen lets say at 3rd print: 12684 samples out of 60000 during this epoch.

Loss: 
- example epoch above shows the avg loss across all test batches during this epoch: 1.51. Accuracy is % of correct predictions. This is evaluated on test or validation dataset.
- It is a measure of how wrong the model's predictions are.
- High loss -> model is making bad predicitions.
- Low loss -> model is learning and improving.
- As training continues, you'd expect the loss to gradually decrease both in these per-batch printouts and in the test set loss at the end of each epoch.

Code: 
Model: 
- defines a neural network with: 
    - input layer for 28x28 = 784 pixels (grayscale image)
    - 2 hidden layers with 512 neurons each
    - Output layer with 10 neurons (one per class so we have 10 classes)

Training:
- In each epoch (full pass through the training dataset):
    - You break the data into batches (size 64 here)
    - For each batch:
        - Run input x through the model -> get predictions (logits)
        - Compare predictions to labels y -> compute loss
        - Use loss.backwared() to calculate gradients
        - Optmizer step updates weights to reduce loss
        - Print output means current loss after steps of training

Testing:
- At the end of each epoch, you evaluate on unseen test data
- Counts how many predictions matched the actual labels and prints accuracy, avg loss.

Prediction:
- After training we get the first image and label from test data
- Feed the image to the model.
- The model picks the index (0-9) with highest score (or predicted class)
- We then map the index to class name in classes[] and pull our predicted label

This line: 
x, y = test_data[INDEX][0], test_data[INDEX][1]
grabs a single image and its associated label from test_data dataset.
This acceses the index-th example in the test dataset(0th, 5th, 100th, etc..)
Each item in test_data is a tuple.
(image tensor, label)
EX: test_data[5] will return something like:
(<28x28 tensor of pixel values>, 2)
Where 2 is the true class index (like "Pullover")
[0][1] is unpacking the tuple:
    - test_data[index][0] -> image (1x28x28 tensor)
    - test_data[index][1] -> label (int from 0-9)

So x, y = test_data[3][0], test_data[3][1] means x -> image and y -> label associated