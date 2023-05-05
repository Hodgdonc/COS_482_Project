import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("metacritic_data_clean.csv")

data.dropna(inplace=True)

# print(data.head(20))

data.plot(x='release_year', y='audience_score', kind='scatter', figsize=(5,3),
        title='movie release year vs. audience score of movies')

plt.show()

data.to_csv("meta_noNaN.csv")