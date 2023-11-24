import numpy as np 
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None

df = pd.read_csv('high_diamond_ranked_10min.csv')

print("DataFrame shape:", df.shape)

missing_values = df.isnull().sum()
print("Missing values:")
print(missing_values)

df.drop(df.columns[[0] + list(range(21, 40))], inplace=True, axis=1)
df.info()

labels = ['Blue wins', 'Red wins']
sizes = [
    df.blueWins[df['blueWins'] == 1].count(),
    df.blueWins[df['blueWins'] == 0].count()
]
explode = [0.1, 0]
colors = ['skyblue', 'lightcoral']  # Define colors here

# We eliminated the RedTeam dataset during our data preprocessing. 
# I am coding a pie graph plot in order to show the similarity of the Winrate of both teams. 
# Red Team data is considered redundant and is not needed for the final result.
plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',  # Use double '%' to display '%'
    pctdistance=0.4,
    shadow=True,
    wedgeprops={'edgecolor': 'black'},
)

plt.title("Win Rates Red and Blue Team", fontweight='bold')
plt.axis('equal')
plt.show()

# Next I'll use the seaborn library to import the data into a heatmap.

plt.figure(figsize=(19,12))
sns.heatmap(df.drop('blueWins' , axis=1).corr(), annot=True, vmin=0)
plt.show()

# After initially running the heatgraph, we can see that blueGoldPerMin and 
# blueGoldTotal have multicollinearity between the two. So we drop one to avoid it.

df = df.drop('blueGoldPerMin', axis=1)

# We want to drop blueWins and blueDeaths because we don't want negative values 
# and we do not need to see blueWins in a win correlation bar plot.

corr = df.corr()["blueWins"].drop(['blueWins' ,'blueDeaths'])
corr = corr.sort_values(ascending=False)
sns.barplot(x=corr, y=corr.index)
plt.show()

# From the bar plot we generated, we can see which elements are outliers.
# Using this information, we can create a scatter matrix to gain more information.

columns = ['blueWardsPlaced', 'blueWardsDestroyed', 'blueDeaths']
pd.plotting.scatter_matrix(df[columns], alpha=.2, figsize=(7, 7), diagonal='kde')
plt.show()
