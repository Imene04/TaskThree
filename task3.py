import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Create a sample DataFrame if 'customer_churn.csv' doesn't exist
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'age': [34, 45, 23, 43, 36],
    'tenure': [5, 10, 2, 8, 6],
    'churn': [0, 1, 0, 0, 1]  # 0 = No Churn, 1 = Churn
}

df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv('customer_churn.csv', index=False)

# Step 2: Load the dataset
data = pd.read_csv('customer_churn.csv')

# Step 3: Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Step 4: Define features (X) and the target variable (y)
X = data.drop('churn', axis=1)  # Replace 'churn' with the actual column name for churn in your dataset
y = data['churn']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Step 7: Train the classifier
clf.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = clf.predict(X_test)

# Step 9: Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(report)
