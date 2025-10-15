import pandas as pd

#load data set
df = pd.read_csv('train.csv')

# --- Inspection Commands ---

# Print the first 5 rows to see what it looks like
print("--- First 5 Rows ---")
print(df.head())

# Get a quick summary of the data types and non-null values
print("\n--- Data Info ---")
print(df.info())

# Get descriptive statistics for numerical columns
print("\n--- Descriptive Stats ---")
print(df.describe())

# Get the dimensions of the DataFrame (rows, columns)
print("\n--- Shape ---")
print(df.shape)

#challenge

#Print first 10 rows
print("--- first 10 rows ---")
print(df.head(10))

#print last 10 rows
print("--- last 10 rows ---")
print(df.tail(10))

# --- Selection and Filtering ---

# Select a single column (this returns a Series)
ages = df['Age']
print("\n--- Age Column (Series) ---")
print(ages.head())

# Select multiple columns (note the double brackets)
subset = df[['Name', 'Sex', 'Age']]
print("\n--- Subset of Columns ---")
print(subset.head())

# Filter rows based on a condition (Boolean Indexing)
# Get all passengers who are older than 60
seniors = df[df['Age'] > 60]
print("\n--- Passengers older than 60 ---")
print(seniors)

# Combine conditions
# Get all female passengers who are older than 60
senior_women = df[(df['Age'] > 60) & (df['Sex'] == 'female')]
print("\n--- Senior Women ---")
print(senior_women)

#challenge

#filter pclass and fare
group = df[["Pclass", "Fare"]]
print(group)

#filter all passengers who embarked from port 'C'
port_c = df[df['Embarked'] == 'C']
print(port_c)

#filter all the male passengers who survived
male_surv = df[(df["Survived"] == 1) & (df["Sex"] == "male")]
print(male_surv.describe())