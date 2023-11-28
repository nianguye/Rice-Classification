import pandas as pd
import torch
import numpy as np
from torch import nn, optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# Basically is the program that we'll use to train the data
# If cuda and backends isn't available, the cpu is used by default
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# The neural network class
# Input Node Size: 7 Input
# Output Node Size: 1 Output (What type of rice it is)
# Hidden/Output Activation: Sigmoid 
# Input: Linear combination
class NeuralNetwork(nn.Module):
    def __init__(self, input_val, hidden_layers):
        super().__init__() # Inheriting the previous initialzied values from nn.Module
        self.input = nn.Linear(input_val, hidden_layers[0]).to(torch.float32)
        # nn.Module is basically python's list that can store modules.
        self.hidden = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]).to(torch.float32) for i in range(len(hidden_layers) - 1)])
        self.output = nn.Linear(hidden_layers[-1], 2).to(torch.float32)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.input(x))
        # Iterate through the hidden layer, where the Linear input (item) passes through the activation function Sigmoid
        for items in self.hidden:
            x = self.sigmoid(items(x))
        x = self.output(x)
        return x



df = pd.read_csv("riceclass.csv")

df_input = df.drop(["id", "Class"], axis = 1)
df_output = df["Class"]

# Apply one hot encoding
oh_encoder = OneHotEncoder()
df_output = pd.get_dummies(df_output, dtype = float)
print(df_output)

# Min Max Scalar
scale = MinMaxScaler(feature_range=(0, 1))
input_rescale = scale.fit_transform(df_input)
df_input = pd.DataFrame(data = input_rescale, columns = df_input.columns)

# 
inputTensor = torch.tensor(df_input.to_numpy(), dtype = torch.float32)
outputTensor = torch.tensor(df_output.to_numpy()).reshape(-1, 2)

# Train Test Split (80% Train)
tensorDf = torch.utils.data.TensorDataset(inputTensor, outputTensor)
train_amount = int(0.8 * len(df))
test_amount = len(df) - train_amount
train_df, test_df = torch.utils.data.random_split(tensorDf, [train_amount, test_amount])

# Set up our Neural network variables
input_size = 7
hidden_layer_size = [12, 24, 48]
learning_rate = 0.01
amount_epochs = 1000

# Create the model ()
model = NeuralNetwork(input_size, hidden_layer_size)

# Set up loss functions for backward propagation and optimizeers
loss = nn.BCEWithLogitsLoss() 
optimize = optim.Adam(model.parameters(), lr = learning_rate)

# Batch SGD

train_data_loader = DataLoader(train_df, batch_size = 300, shuffle = True)

for epochs in range(1):
    for input_val, output_val in train_data_loader:
        optimize.zero_grad()
        # Use the model with the input data, where result are the outputs.
        result = model(input_val)
        # Back propagation with optimizer
        # view transposes the output_val for comparison
        lossResult = loss(result, output_val)
        lossResult.backward()
        optimize.step()

    # Print progress
    if (epochs + 1) % 100 == 0:
        print(f'Epoch [{epochs+1}/{amount_epochs}], Loss: {lossResult.item():.4f}')


# Comparison with test data
model.eval() # PyTorch  
test_data_loader = DataLoader(test_df, batch_size = 300, shuffle = True)

correct_ans = 0
total_ans = 0
for input_val, output_val in test_data_loader:
    # Use the model with the input data, where result are the outputs.
    result = model(input_val)
    print(result)
    round_result = torch.round(torch.sigmoid(result)) # Rounds the answer so that it goes to 0 or 1 for MSE analysis
    total_ans += output_val.size(0) * 2 # * 2 since there's 2 columns, so the denominator has to be doubled
    # item() converts from torch to python integer
    correct_ans += (round_result == output_val).sum().item()
print("Accuracy: " + str(correct_ans/total_ans * 100))


# Hypertuning
# Using skorch as a wrapper to convert Pytorch to scikit-learn

scikit_model = NeuralNetClassifier(
    module=NeuralNetwork,
    module__input_val = input_size,
    max_epochs = 1000,
    batch_size = 300,
    criterion= nn.BCEWithLogitsLoss(),
    iterator_train__shuffle=True
)

# The parameter that's getting hypertune (This project doesn't require the input layer to be hypertuned since the dataset assumes all parameters are necessary)

param_grid = {
    'optimizer': [optim.SGD, optim.Adam],
    'optimizer__lr': [ 0.01, 0.001, 0.0001, 0.1],
    'module__hidden_layers': [[128, 128, 128], [32, 32],[12,24,48], [7,7,7], [32, 128, 32]]
}

grid = GridSearchCV(estimator=scikit_model, param_grid=param_grid, cv=10, scoring ='accuracy', n_jobs = -1)
# Convert the datatype for df_input and df_output to numpy float 32
grid_result = grid.fit(inputTensor, y = outputTensor)

# The best parameters to use.
print(str(grid_result.best_params_))
