import tensorflow as tf
import numpy as np
from numpy import loadtxt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### load required datasets
train_data = loadtxt("imdb_NN_Ready_gross.csv", delimiter = ',', skiprows=1)
## split into input (x) and output (y)
train_x = train_data[:, :-1]   # we'll be using all but the last one as inputs
train_y = train_data[:, -1:]   # to determine the final one (gross)

## scale features to standard normal
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

x, xt, y, yt = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

### building the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(x.shape[1],), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)    # 1 node for output (gross)
])

### compiling the model
model.compile(optimizer="adam", loss="mse")

### train the model
model.fit(x, y, epochs=300, batch_size=256, validation_data=(xt, yt))

### evaluate the accuracy of the model by running it on the test data
test_loss = model.evaluate(xt, yt, verbose=0) 
print("loss:", test_loss)

## takes a model, training inputs (x), training outputs(y), test inputs(xt), and
## test outputs(yt) and calculates the average loss(MSE)
def avgRuns(model, x, y, xt, yt, runs):
    lossList = []

    for i in range(runs):
        modeli = tf.keras.models.clone_model(model)
        modeli.compile(optimizer="adam", loss="mse")
        modeli.fit(x, y, epochs=300, batch_size=256, validation_data=(xt, yt))

        loss = modeli.evaluate(xt, yt, verbose=0)

        lossList.append(loss)

    print()
    print(runs, " run average:")
    print('Loss: %.3f (%.3f)' % (np.mean(lossList), np.std(lossList)))

avgRuns(model, x, y, xt, yt, 10)

### attempt to predict the amnt grossed of the test data
### for the purposes of visually comparing prediction to the test data
### i.e predict yt by inputting xt

prediction = model.predict(xt)

prediction_df = pd.DataFrame(prediction, columns = ["predicted_grossing"])

## read in test data as dataframe to compare to
correct_df = pd.DataFrame(yt, columns = ["gross_US_CA"])

## concatenate dfs together
comparison_df = pd.concat([correct_df, prediction_df], axis = 1)

## calculate percent difference of gross columns
comparison_df["percent_gross_diff"] = (100 * abs(comparison_df["gross_US_CA"] -
                                                comparison_df["predicted_grossing"])
                                                / comparison_df["gross_US_CA"])

comparison_df.to_csv("imdb_gross_prediction_comparison.csv", index=False)