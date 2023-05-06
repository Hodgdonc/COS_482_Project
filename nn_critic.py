import tensorflow as tf
import numpy as np
from numpy import loadtxt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_data = loadtxt("meta_nn_ready.csv", delimiter = ',', skiprows=1)
## split into input (x) and output (y)
train_x = train_data[:, :-1]   
train_y = train_data[:, -1:]   

x, xt, y, yt = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

## build model
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

## predict and store diff in csv
prediction = model.predict(xt)

prediction_df = pd.DataFrame(prediction, columns = ["predicted_critic"])

## read in test data as dataframe to compare to
correct_df = pd.DataFrame(yt, columns = ["critic"])

## concatenate dfs together
comparison_df = pd.concat([correct_df, prediction_df], axis = 1)

## calculate percent difference of gross columns
comparison_df["percent_diff"] = (100 * abs(comparison_df["critic"] -
                                                comparison_df["predicted_critic"])
                                                / comparison_df["critic"])

comparison_df.to_csv("meta_predicted_critic.csv", index=False)