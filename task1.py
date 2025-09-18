# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# (Replace with actual dataset path or URL)
data = pd.read_csv("house_price_dataset.csv")

# Show first few rows
print(data.head())

# Features (independent variables)
X = data[['sqft', 'bedrooms', 'bathrooms']]   # make sure column names match dataset

# Target (dependent variable)
y = data['price']

# Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Model performance
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Example prediction
example = pd.DataFrame({
    'sqft': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})
predicted_price = model.predict(example)
print("Predicted Price for example house:", predicted_price[0])