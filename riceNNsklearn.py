import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score

df = pd.read_csv("riceclass.csv")

df_input = df.drop(["id", "Class"], axis = 1)
df_output = df["Class"]

scale = MinMaxScaler(feature_range=(0, 1))
input_rescale = scale.fit_transform(df_input)
df_input = pd.DataFrame(data = input_rescale, columns = df_input.columns)

df_output = pd.get_dummies(df_output, dtype = int)

input_train, input_test, output_train, output_test = train_test_split(df_input, df_output, test_size=0.2)
mlp = MLPClassifier(solver = 'adam', random_state = 42, activation = 'logistic', learning_rate_init = 0.01, batch_size = 300, hidden_layer_sizes = (12, 24, 48,), max_iter = 1000)
mlp

mlp.fit(input_train, output_train)

predict = mlp.predict(input_test)
predict

param_grid = dict(
    learning_rate_init = [0.01, 0.001, 0.3, 0.1, 0.0001],
    solver = ['sgd','adam'],
    hidden_layer_sizes = [(128, 128, 128,), (32, 32,),(12,24,48,), (12,3), (32, 128, 32,)]
)
print("Accuracy: ", accuracy_score(output_test, predict))
print("Mean Square Error : ", mean_squared_error(output_test, predict))

grid = GridSearchCV(estimator = mlp, param_grid = param_grid)
#grid.fit(input_train, output_train)

print("Optimal Hyper-parameters : ", grid.best_params_)
print("Optimal Accuracy : ", grid.best_score_)