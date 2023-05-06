import pandas as pd

data = pd.read_csv("meta_noNaN.csv")

del data["release_date"]
del data["title"]

one_hot_encoded_rating = pd.get_dummies(data['rating'], prefix='rated').astype(int)
data = pd.concat([data, one_hot_encoded_rating], axis=1)

one_hot_encoded_year = pd.get_dummies(data['release_year'], prefix='released').astype(int)
data = pd.concat([data, one_hot_encoded_year], axis=1)

## remove old versions
del data["release_year"]
del data["rating"]

columns = list(data.columns)
columns.remove("audience_score")
columns.remove("critic_score")
columns.append("audience_score")
columns.append("critic_score")
data = data.reindex(columns=columns)

del data["Unnamed: 0"]
del data["Unnamed: 0.1"]

## convert audience and critic score to ranges between 0 and 1 so that
## we can plug straight into sigmoid
data["audience_score"] = data["audience_score"] / 10.0
data["critic_score"] = data["critic_score"] / 100.0

data.to_csv("meta_nn_ready.csv", index = False)

columns.remove("audience_score")
columns.append("audience_score")
data = data.reindex(columns=columns)

del data["Unnamed: 0"]
del data["Unnamed: 0.1"]

data.to_csv("meta_nn_ready_audience.csv", index = False)