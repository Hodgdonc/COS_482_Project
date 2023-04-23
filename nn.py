import tensorflow as tf
import numpy as np
from numpy import loadtxt
import pandas as pd

### load required datasets
train_data = loadtxt("imdb_train.csv", delimiter = ',', skiprows=1)
## split into input (x) and output (y)
x = train_data[:,0:5]   # we'll be using the first 5 of each row
y = train_data[:,5:7]   # to determine the final two (score and amnt grossed)

test_data = loadtxt("imdb_test.csv", delimiter=',', skiprows=1)
xt = test_data[:,0:5]
yt = test_data[:,5:7]

### building the model
model = tf.keras.Sequential([
    ## input shape is 5 because each row has 5 inputs
    tf.keras.layers.Dense(128, input_shape=(5,), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    ## number of nodes (128, 64, 32) and layers determined by playing around, 
    ## seeing how accuracy changed - yields accuracy above 60% usually
    ## TODO see if seperating train and test by alternating rows instead
    ## of splitting str8 in half will improve further
    tf.keras.layers.Dense(2)    # 2 nodes for output (score and amnt grossed)
])

### compiling the model
## loss function chosen according to https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])

### train the model
## epochs and batch size according to https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model
## and also from playing around with the numbers
model.fit(x, y, epochs=300, batch_size=256)

### evaluate the accuracy of the model by running it on the test data
test_loss, test_acc = model.evaluate(xt, yt, verbose=0) 
test_accP = f'{test_acc:.6%}'
print('\nTest accuracy:', test_accP)

### attempt to predict the score and amnt grossed of the test data
### for the purposes of visually comparing prediction to the test data
### i.e predict yt by inputting xt
prediction = model.predict(xt)
## TODO: seperate this model into two seperate ones, one for score and one for 
## grossing so that I can properly limit the range on both seperately.
## probably makes more sense that way too, considering grossing is probably the
## best way to predict its score and vice versa
## prediction = np.clip(preprediction, a_min=0.0, a_max=10.0)
prediction_df = pd.DataFrame(prediction, columns = ["predicted_grossing", 
                                                    "predicted_score"])

## read in test data as dataframe to compare to
correct_df = pd.read_csv("imdb_test.csv")

## concatenate dfs together
comparison_df = pd.concat([correct_df, prediction_df], axis = 1)

## calculate percent difference of grossing and score columns
comparison_df["percent_gross_diff"] = (100 * abs(comparison_df["gross_US_CA"] -
                                                comparison_df["predicted_grossing"])
                                                / comparison_df["gross_US_CA"])

comparison_df["percent_score_diff"] = (100 * abs(comparison_df["imdb_score"] -
                                                comparison_df["predicted_score"])
                                                / comparison_df["imdb_score"])

comparison_df.drop(["release_year","length","genre","rating","imdb_votes"], 
                   axis=1, inplace=True)

comparison_df.to_csv("prediction_comparison.csv", index=False)