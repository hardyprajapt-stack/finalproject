# ------------------------------------------------------------
# TITANIC SURVIVAL ANALYSIS PROJECT
# ------------------------------------------------------------

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Step 2: Load the dataset
data = pd.read_csv(r"C:\Users\Education\Downloads\titanic.csv")

# ------------------------------------------------------------
# Step 3: View basic information about the dataset
print("Dataset Overview:")
print(data.head())
print(data.info())
print(data.describe())

# ------------------------------------------------------------
# Step 4: Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# ------------------------------------------------------------
# Step 5: Fill missing values
data['Age'] = data['Age'].fillna(data['Age'].median())

data['Embarked'] = data['Embarked'].fillna(
    data['Embarked'].mode()[0]
)

data.drop('Cabin', axis=1, inplace=True)

# ------------------------------------------------------------
# Step 6: Verify cleaning
print("\nAfter Cleaning:")
print(data.isnull().sum())

# ------------------------------------------------------------
# Step 7: Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()

# ------------------------------------------------------------
# Step 8: Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("Survival by Gender")
plt.show()

# ------------------------------------------------------------
# Step 9: Survival by Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival by Passenger Class")
plt.show()

# ------------------------------------------------------------
# Step 10: Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# ------------------------------------------------------------
# Step 11: Age Group Analysis
data['AgeGroup'] = pd.cut(
    data['Age'],
    bins=[0,12,18,35,60,100],
    labels=['Child','Teen','Young Adult','Adult','Senior']
)

plt.figure(figsize=(8,5))
sns.countplot(x='AgeGroup', hue='Survived', data=data)
plt.title("Survival by Age Group")
plt.show()

# ------------------------------------------------------------
# Step 12: Correlation Heatmap
numeric_data = data.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# Step 13: Insights
print("\nKey Insights:")
print("1. Females survived more than males.")
print("2. 1st class passengers survived more.")
print("3. Children had better survival chances.")

# ------------------------------------------------------------
# Step 14: Save cleaned dataset
data.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ------------------------------------------------------------
# END