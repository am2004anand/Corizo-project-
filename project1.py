import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("cardio_train.csv", delimiter=";") 

#data preprocessing
# Display dataset information
print("Dataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop 'id' column if it exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

df_sample = df.sample(n=10000, random_state=42) 

# Separate features (X) and target (y)
X = df_sample.drop("cardio", axis=1)
y = df_sample["cardio"]

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Data Analysis and Visualizations
# Univariate Analysis: Histograms for all features
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Features")
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Features")
plt.show()

# Comparing Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_jobs=-1),
    "SVM (Linear Kernel)": SVC(kernel="linear")
}

# Train and Evaluate Models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    results[name] = acc
    print(f"{name} Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Comparing Model Accuracies
result_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
result_df.sort_values(by="Accuracy", ascending=False, inplace=True)
print("\nModel Performance Comparison:")
print(result_df)

# Build the Best Model (Example: Random Forest)
print("\nBuilding Final Model (Random Forest)...")
best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
final_predictions = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)

print("\nFinal Model Performance:")
print(f"Random Forest Accuracy: {final_accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_predictions))
print("Classification Report:")
print(classification_report(y_test, final_predictions))
