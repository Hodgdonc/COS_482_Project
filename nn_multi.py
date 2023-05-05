import tensorflow as tf
import numpy as np
from numpy import loadtxt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### load required datasets
train_data = loadtxt("imdb_NN_Ready.csv", delimiter = ',', skiprows=1)
## split into input (x) and output (y)
train_x = train_data[:, :-2]   # we'll be using all but the last two as inputs
train_y = train_data[:, -2:]   # to determine the final two (score and amnt grossed)

## scale features to standard normal
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

x, xt, y, yt = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

### building the model - multi output regression
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(x.shape[1],), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    ## number of nodes (128, 64, 32) and layers determined by playing around, 
    ## seeing how accuracy changed - yields accuracy above 60% usually
    tf.keras.layers.Dense(2)    # 2 nodes for output (score and amnt grossed)
])

### compiling the model
model.compile(optimizer="adam", loss="mse" , metrics="accuracy")

### train the model
## epochs and batch size according to https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model
## and also from playing around with the numbers
model.fit(x, y, epochs=300, batch_size=256, validation_data=(xt, yt))

### evaluate the accuracy of the model by running it on the test data
test_loss, test_acc = model.evaluate(xt, yt, verbose=0) 
test_accP = f'{test_acc:.6%}'
prediction = model.predict(xt)
## mae between predicted gross and actual gross
mse_1 = mean_squared_error(yt[:, 0], prediction[:, 0])
## mae between predicted score and actual score
mse_2 = mean_squared_error(yt[:, 1], prediction[:, 1])
print('\nTest accuracy:', test_accP)
print("loss:", test_loss)
print("mse for gross and score: ", mse_1, mse_2)

## takes a model, training inputs (x), training outputs(y), test inputs(xt), and
## test outputs(yt) and calculates the average loss(MSE), accuracy, and 
## mean squared error for both outputs
def avgRuns(model, x, y, xt, yt, runs):
    lossList = []
    accList = []
    mse1List = []
    mse2List = []

    for i in range(runs):
        modeli = tf.keras.models.clone_model(model)
        modeli.compile(optimizer="adam", loss="mse" , metrics="accuracy")
        modeli.fit(x, y, epochs=300, batch_size=256, validation_data=(xt, yt))

        loss, acc = modeli.evaluate(xt, yt, verbose=0)
        prediction = modeli.predict(xt)
        mse_1 = mean_squared_error(yt[:, 0], prediction[:, 0])
        mse_2 = mean_squared_error(yt[:, 1], prediction[:, 1])

        lossList.append(loss)
        accList.append(acc)
        mse1List.append(mse_1)
        mse2List.append(mse_2)

    print()
    print(runs, " run average:")
    print('Accuracy: %.3f (%.3f)' % (np.mean(accList), np.std(accList)))
    print('Loss: %.3f (%.3f)' % (np.mean(lossList), np.std(lossList)))
    print('MSE for gross: %.3f (%.3f)' % (np.mean(mse1List), np.std(mse1List)))
    print('MSE for score: %.3f (%.3f)' % (np.mean(mse2List), np.std(mse2List)))

# avgRuns(model, x, y, xt, yt, 10)


## TODO: seperate this model into two seperate ones, one for score and one for 
## grossing so that I can properly limit the range on both seperately.

### attempt to predict the score and amnt grossed of the test data
### for the purposes of visually comparing prediction to the test data
### i.e predict yt by inputting xt
prediction_df = pd.DataFrame(prediction, columns = ["predicted_grossing", 
                                                    "predicted_score"])

## read in test data as dataframe to compare to
correct_df = pd.DataFrame(yt, columns = ["gross_US_CA", "imdb_score"])

## concatenate dfs together
comparison_df = pd.concat([correct_df, prediction_df], axis = 1)

## calculate percent difference of grossing and score columns
comparison_df["percent_gross_diff"] = (100 * abs(comparison_df["gross_US_CA"] -
                                                comparison_df["predicted_grossing"])
                                                / comparison_df["gross_US_CA"])

comparison_df["percent_score_diff"] = (100 * abs(comparison_df["imdb_score"] -
                                                comparison_df["predicted_score"])
                                                / comparison_df["imdb_score"])

comparison_df.to_csv("imdb_multi_prediction_comparison.csv", index=False)