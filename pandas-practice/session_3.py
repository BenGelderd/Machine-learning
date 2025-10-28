import pandas as pd 

# Load the data
df = pd.read_csv('train.csv')

print("--- GroupBy Examples ---")

# Example 1: What was the average age of passengers in each class?
# 1. Group by 'Pclass'
# 2. Select the 'Age' column
# 3. Calculate the mean
avg_age_by_class = df.groupby('Pclass')['Age'].mean()
print("\nAverage age per class:")
print(avg_age_by_class)

# Example 2: How many people survived vs. died, grouped by sex?
# This is a more advanced aggregation.
survival_stats = df.groupby('Sex').agg(
    total_passengers=('PassengerId', 'count'),  # Count passengers in each group
    total_survived=('Survived', 'sum')          # Sum the 'Survived' column (since 1=survived, 0=died)
)
print("\nSurvival stats by sex:")
print(survival_stats)

"""
Challenge:
Find the average Fare paid by passengers grouped by their embarkation port (Embarked).

Find the highest (.max()) and lowest (.min()) age for each passenger class (Pclass).
"""

print("---CHALLENGE---")
avg_fare_paid = df.groupby('Embarked')['Fare'].mean()
print("\nAverage Fare by port")
print(avg_fare_paid)

class_ages = df.groupby('Pclass').agg(
    max_height = ('Age', 'max'),
    min_height = ('Age', 'min')
)
print("\nMax and Min height by class")
print(class_ages)

"""
.merge() practice:
real-world data is almost never in one clean file. You might have one file with passenger 
info and another file with details about the ship's cabins. pd.merge() is the tool 
(just like a SQL JOIN) to combine them.

"""

# Create a new, small DataFrame with details about each class
class_details_data = {
    'Pclass': [1, 2, 3],
    'Class_Name': ['First Class', 'Second Class', 'Third Class'],
    'On_Deck': ['A', 'C', 'F']
}
df_class_details = pd.DataFrame(class_details_data)

print("\n--- Our new Class Details table ---")
print(df_class_details)

# Merge the main df with df_class_details
# We tell it to match rows based on the 'Pclass' column
# 'how='left'' means: "Keep every passenger from the left (main)
# dataframe, and add info from the right one where it matches."
df_merged = pd.merge(df, df_class_details, on='Pclass', how='left')

print("\n--- Merged DataFrame ---")
# See the new 'Class_Name' and 'On_Deck' columns at the end!
print(df_merged.head())

"""
Challenge:
How would you check the Fare of only the passengers in 'First Class' using the 
new df_merged DataFrame? (Hint: You'll use the filtering you learned in Session 1).
"""

print("---CHALLENGE---")
Fares_First = df_merged[(df_merged['Class_Name'] == 'First Class')]
print("\nFirst class fares")
print(Fares_First['Fare'])