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

# Using the information gained from this scatter matrix, we can now count the occurence 
# of outliers.

print(f"More than 120 wards placed: {len(df[df['blueWardsPlaced'] >= 120])} games")
print(f"More than 17 Wards destroyed: {len(df[df['blueWardsDestroyed'] >= 17])} games")
print(f"More than 16 deaths: {len(df[df['blueDeaths'] >= 16])} games")

# This data gives us really important information. In order for 120 wards to be placed in 
# the first 10 minutes of the game, each plaer would have to place 2 wards a minute, which is impossible
# due to the cooldown to place a ward being 90 seconds. So we should exclude this abnormality. The same reason
# will be used for Wards Destroyed due to the Oracle (Item used to discover and destroy wards) cooldown being 
# 120 seconds. Games having above 16 deaths in 10 minutes should almost never happen at this high of a skill level.
# If this amount of deaths occured then it's pretty safe to assume a player was intentionally throwing the game. Therefore
# the player was not trying, so the data is not reliable.

# Define conditions to filter outliers
condition_wards_placed = df['blueWardsPlaced'] < 120
condition_wards_destroyed = df['blueWardsDestroyed'] < 17

# Apply conditions to the DataFrame
df_filtered = df[condition_wards_placed & condition_wards_destroyed]

# Calculate the number of removed outliers
removed_outliers = df.shape[0] - df_filtered.shape[0]
print(f'Removed {removed_outliers} outliers')

if len(df_filtered) > 0:
    print(df_filtered.head(5))
else:
    print("No rows satisfy both conditions.")


