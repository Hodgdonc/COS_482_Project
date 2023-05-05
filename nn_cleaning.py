import pandas as pd

data = pd.read_csv("movies_cleaned_onehot_genres_no_NaN.csv")

del data["Unnamed: 0"]  # remove extraneous ID column
del data["title"]       # remove movie title (doesn"t make sense as categorical)
del data["director"]    # same reason ^
del data["genre"]       # we'll use the hot encoded versions

## convert categorical variables into numerical (genre, rating, year)
one_hot_encoded_rating = pd.get_dummies(data['rating'], prefix='rated').astype(int)
data = pd.concat([data, one_hot_encoded_rating], axis=1)

one_hot_encoded_year = pd.get_dummies(data['release_year'], prefix='released').astype(int)
data = pd.concat([data, one_hot_encoded_year], axis=1)

## remove old versions
del data["release_year"]
del data["rating"]

### rearrange columns so desired NN outputs appear last 
## (unneeded, but looks nicer / easier to work with)
columns = list(data.columns)
columns.remove("imdb_score")
columns.remove("gross_US_CA")
columns.append("gross_US_CA")
columns.append("imdb_score")
data = data.reindex(columns=columns)

### export dataframes to csv's for storage
data.to_csv("imdb_NN_Ready.csv", index=False)

## make a new csv with amount grossed at the end for ease of access

columns.remove("gross_US_CA")
columns.append("gross_US_CA")
data = data.reindex(columns=columns)

data.to_csv("imdb_NN_Ready_gross.csv", index=False)