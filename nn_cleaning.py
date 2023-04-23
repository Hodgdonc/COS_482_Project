import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("imdb_movies_FINAL_NO_NAN_CLEANED.csv")

del data["Unnamed: 0"]  # remove extraneous ID column
del data["title"]       # remove movie title (doesn"t make sense as categorical)
del data["director"]    # same reason ^

### rearrange columns so desired NN outputs appear last 
## (unneeded, but looks nicer / easier to work with)
columns = list(data.columns)
columns.remove("imdb_score")
columns.remove("gross_US_CA")
columns.append("gross_US_CA")
columns.append("imdb_score")
data = data.reindex(columns=columns)

### split values in genre column on commas and take first genre only
data["genre"] = data["genre"].str.split(",").str[0]

### remove commas from imdb_votes column
data["imdb_votes"] = data["imdb_votes"].str.replace(",", "")

### convert columns to the proper type (int or float)
## regex statement to eliminate roman numerals that snuck into year column
data['release_year'] = data['release_year'].str.replace(r'^I\)\s*\(|^II\)\s*\(|^IX\)\s*\(|^III\)\s*\(|^VI\)\s*\(|^V\)\s*\(|^IV\)\s*\(|^VII\)\s*\(', '', regex=True)
data["release_year"] = data["release_year"].astype(int)

## votes to ints
data["imdb_votes"] = data["imdb_votes"].astype(int)

## convert categorical variables into numerical (genre, rating)
encoder = LabelEncoder()
data["genre"] = encoder.fit_transform(data["genre"])
data["rating"] = encoder.fit_transform(data["rating"])

### seperate data into training and testing data - half each
## TODO Probably better to take every other line for each, since its sorted by
## score
mid = len(data) // 2
train_data = data.iloc[:mid]
test_data = data.iloc[mid:]

### export dataframes to csv's for storage
data.to_csv("imdb_NN_Ready.csv", index=False)
train_data.to_csv("imdb_train.csv", index=False)
test_data.to_csv("imdb_test.csv", index=False)