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


# We now want to examine the variables that will have significant impact on the end result 
# of the game. We seperate them into two categories : Numerical and Categorical. Numerical would
# would include categories such as blueGoldDiff, blueExperienceDiff, blueKills and blueTotalMInionsKilled.
# While categorial would include variables for Elite monsters killed which are blueDragons and blueHerals.
# But for the Categorial values there are only two values possible, 0 or 1. This is due to the fact the first 
# spawn for the dragon and the herald take over 5 minutes and the next spawn will take an equal amount of time, 
# disallowing for the slaying of more than one dragon or herald.
colors = {
    
    'red' : '#ff9999' ,
    'blue' : '#66b3ff'
    
    
}
custom_palette = sns.color_palette(list(colors.values()))

fig, _ax = plt.subplots(nrows=2, ncols=2, figsize=(9,7))
plt.subplots_adjust(wspace=0.5)

sns.boxplot(data=df_filtered, x='blueWins' , y='blueTotalMinionsKilled' , hue ='blueWins' , palette=custom_palette, ax=_ax[0][0] , legend=False)
sns.boxplot(data=df_filtered, x='blueWins' , y='blueGoldDiff' , hue='blueWins' , palette=custom_palette, ax=_ax[0][1] , legend=False)
sns.boxplot(data=df_filtered, x='blueWins' , y='blueExperienceDiff' , hue='blueWins' , palette=custom_palette, ax=_ax[1][0] , legend=False)
sns.boxplot(data=df_filtered, x='blueWins' , y='blueKills' , hue='blueWins' , palette=custom_palette, ax=_ax[1][1] , legend=False)

plt.show()

# These graphs show us that the team with a lead in Gold, Experience and Kills are more likely to win. But we also 
# want to show that Epic Monster kills also will contribute to wins

fig, _ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
plt.subplots_adjust(wspace=0.5)
sns.countplot(data=df_filtered, x='blueDragons' , hue='blueWins' , palette=custom_palette , ax=_ax[0])
sns.countplot(data=df_filtered, x='blueHeralds' , hue='blueWins' , palette=custom_palette , ax=_ax[1])

plt.show()

# These bar graphs show that blue team tends to win more games if they get the first dragon or herald.

# We now have the features that we are confident are crucial to the output of the game. In order to be more accurate
# we can derive an extra feature from blueKills. We can use KDA(Kills + Assists)/Deaths. This metric is very useful
# determining any players effectivess in any PvP (Player vs Player) game. 

mask = df_filtered['blueDeaths'] == 0
df_filtered = df_filtered.copy() 
df_filtered.loc[mask, 'kda'] = (df_filtered['blueKills'] + df_filtered['blueAssists']) / 0.5
df_filtered.loc[~mask, 'kda'] = (df_filtered['blueKills'] + df_filtered['blueAssists']) / df_filtered['blueDeaths']
