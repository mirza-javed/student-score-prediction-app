# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib


# Load the dataset
df = pd.read_csv('exams.csv')

# Quick Check of the dataset
# print(df.shape)
# print(df.columns)

# Creating a new column for average score
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3

# one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['gender', 'lunch', 'test preparation course'], drop_first=True)
# Ordinal encoding for parental level of education
education_order = {
    'some high school' : 0,
    'high school' : 1,
    'some college' : 2,
    "associate's degree" : 3,
    "bachelor's degree" : 4,
    "master's degree" : 5
}
df['parental level of education'] = df['parental level of education'].map(education_order)

# Convert True/False columns to proper 1/0 integers for safety
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Dropping unnecessary columns
df = df.drop(columns=['race/ethnicity', 'math score', 'reading score', 'writing score'])

# Splitting the dataset into features and target variable
X = df.drop(columns=['average_score'])
y = df['average_score']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Training the Linear Regression Model
pipeline = Pipeline([
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)

# Making predictions on the test set
y_pred_lr = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2 = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - MAE: {mae}, RMSE: {rmse}, R²: {r2}")

# Save the trained pipeline (includes the fitted LinearRegression model)
joblib.dump(pipeline, 'student_score_model.pkl')
print("Model saved successfully!")

import matplotlib.pyplot as plt

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='teal')

# Add the "perfect prediction" reference line (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Linear Regression: Actual vs Predicted Scores")
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)  # saves the image to your project folder
plt.show()

# Extract the trained LinearRegression object from the pipeline
lr_model = pipeline.named_steps['regressor']
# Reuse your coefficients DataFrame from earlier
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient')  # sort so the chart reads cleanly, most negative to most positive

plt.figure(figsize=(8, 5))
colors = ['crimson' if c < 0 else 'seagreen' for c in coefficients['Coefficient']]
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)

plt.xlabel("Coefficient Value (Effect on Predicted Score)")
plt.title("Linear Regression Feature Coefficients")
plt.axvline(x=0, color='black', linewidth=0.8)  # zero-reference line
plt.tight_layout()
plt.savefig('feature_coefficients.png', dpi=150)
plt.show()