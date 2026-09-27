"""
Task 6 - Deliverable 2: Feature Engineering Implementation
Data Science with Python Internship | Trisha S Shetty (USSN U06NE24S0054)

Standalone script that performs all feature-engineering steps on the Titanic
dataset and saves the engineered dataset to CSV for review.
"""

import pandas as pd

# ---------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------
df = pd.read_csv("titanic.csv")
data = df.copy()

# ---------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# ---------------------------------------------------------------
# 3. FamilySize & IsAlone
# ---------------------------------------------------------------
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

# ---------------------------------------------------------------
# 4. Title extraction from Name (optional feature from brief)
# ---------------------------------------------------------------
data["Title"] = data["Name"].str.extract(r",\s*([^\.]*)\.")
rare_titles = data["Title"].value_counts()[data["Title"].value_counts() < 10].index
data["Title"] = data["Title"].replace(rare_titles, "Rare")
data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# ---------------------------------------------------------------
# 5. Age groups (optional feature from brief)
# ---------------------------------------------------------------
data["AgeGroup"] = pd.cut(
    data["Age"], bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)

# ---------------------------------------------------------------
# 6. Encode categorical columns
# ---------------------------------------------------------------
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data_encoded = pd.get_dummies(data, columns=["Embarked", "Title", "AgeGroup"], drop_first=True)

# ---------------------------------------------------------------
# 7. Final feature set
# ---------------------------------------------------------------
feature_cols = [c for c in data_encoded.columns if c not in
                ["PassengerId", "Survived", "Name", "Ticket", "Cabin"]]

print("Raw columns:", list(df.columns))
print()
print("New engineered features added: FamilySize, IsAlone, Title, AgeGroup")
print()
print(f"Final feature set ({len(feature_cols)} features):")
print(feature_cols)

# ---------------------------------------------------------------
# 8. Save engineered dataset
# ---------------------------------------------------------------
data_encoded.to_csv("engineered_titanic_dataset.csv", index=False)
print("\nSaved: engineered_titanic_dataset.csv")
