import numpy as np
from numpy import loadtxt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#### establishes basline mean squared error for imdb dataset score and amnt gross

### load required datasets
train_data = loadtxt("imdb_NN_Ready.csv", delimiter = ',', skiprows=1)
train_x = train_data[:, :-2]   
train_y = train_data[:, -2:]  
x, xt, y, yt = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

## means of target variables in training set
mean_target1 = y[:, 0].mean()       # mean of amount grossed
mean_target2 = y[:, 1].mean()       # mean of score

## use mean values to predict target vars in test set
y_pred_1 = [mean_target1] * len(yt)
y_pred_2 = [mean_target2] * len(yt)
y_pred = np.column_stack((y_pred_1, y_pred_2))      # all preds

## compute mean squared error between predicted values for test set and
## actual values for test set
mse_1 = mean_squared_error(yt[:, 0], y_pred_1)
mse_2 = mean_squared_error(yt[:, 1], y_pred_2)
mse = np.array([mse_1, mse_2])      # all mse's

# Print the baseline MSE for each target variable
print("Baseline MSE for target 1:", mse_1)
print("Baseline MSE for target 2:", mse_2)
print("overall MSE: ", (mse_1+mse_2)/2)


