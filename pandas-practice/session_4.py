import pandas as pd
import numpy as np # We'll need this

# Load the data
df = pd.read_csv('train.csv')

print("--- Handling Missing Data ---")

# 1. How to find missing data?
# This shows a count of missing values for each column.
print("Missing values before cleaning:")
print(df.isnull().sum())

# 2. Strategy 1: Dropping (Not always a good idea)
# Let's drop the 'Cabin' column entirely since it's mostly empty
# 'axis=1' means we are dropping a column, not a row
df_dropped = df.drop('Cabin', axis=1)

# 3. Strategy 2: Filling (Imputing)
# This is a much better strategy for 'Age'.
# Let's fill all missing 'Age' values with the average (mean) age.
mean_age = df_dropped['Age'].mean()
# 'inplace=True' modifies the df_dropped directly, no need to re-assign it.
df_dropped['Age'].fillna(mean_age, inplace=True)

print("\nMissing values after cleaning:")
print(df_dropped.isnull().sum())

"""
Challenge: The 'Embarked' column has 2 missing values. 
Instead of filling with a number, you should fill it with the most common value (the "mode").
Can you figure out how to do that? (Hint: df_dropped['Embarked'].mode()[0]).
"""
print("\n---CHALLENGE---")
mode_embarked = df_dropped['Embarked'].mode()[0]

df_dropped['Embarked'].fillna(mode_embarked, inplace=True)
print(df_dropped.isnull().sum())

print("\n--- Feature Engineering ---")

# 1. Create 'FamilySize' from 'SibSp' and 'Parch'
# We add +1 for the passenger themselves
df_dropped['FamilySize'] = df_dropped['SibSp'] + df_dropped['Parch'] + 1

# 2. Create 'IsAlone' based on 'FamilySize'
# Use numpy's .where() function: np.where(condition, value_if_true, value_if_false)
df_dropped['IsAlone'] = np.where(df_dropped['FamilySize'] == 1, 1, 0) # 1 if alone, 0 if not

# 3. Create 'Title' by extracting from the 'Name' column (Advanced)
# This uses .apply() with a function. It's a powerful pattern.
def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

df_dropped['Title'] = df_dropped['Name'].apply(extract_title)

print("\nDataFrame with new features (see new columns at the end):")
print(df_dropped.head())

print("\nDifferent titles found:")
print(df_dropped['Title'].value_counts())