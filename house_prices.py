# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Display first few rows
print("Training Data Preview:")
print(train_data.head())

# Step 3: Data Preprocessing
# Fill missing values only for numerical columns
train_data.fillna(train_data.select_dtypes(include=['number']).mean(), inplace=True)
test_data.fillna(test_data.select_dtypes(include=['number']).mean(), inplace=True)

# Select relevant features (More Features Added)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'YearRemodAdd', 'LotArea']
X_train = train_data[features]
y_train = np.log(train_data['SalePrice'])  # Log Transform SalePrice
X_test = test_data[features]

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Split Data for Model Training and Validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Step 6: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_split, y_train_split)

# Best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Use the best estimator for predictions
best_model = grid_search.best_estimator_

# Step 7: Evaluate the Best Model
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error on Validation Data: {mse}")
print(f"R-squared on Validation Data: {r2}")

# Step 8: Create Actual vs Predicted Table
# Create a DataFrame for Actual vs Predicted
actual_vs_predicted = pd.DataFrame({
    'Actual': np.exp(y_val),  # Reverse log transformation for actual values
    'Predicted': np.exp(y_pred)  # Reverse log transformation for predicted values
})

# Display the Actual vs Predicted table
print("\nActual vs Predicted Table:")
print(actual_vs_predicted.head())

# Step 9: Plot Actual vs Predicted Graph
plt.figure(figsize=(8, 6))

# Scatter plot for Actual vs Predicted values
plt.scatter(np.exp(y_val), np.exp(y_pred), color='blue', alpha=0.5)

# Add a line for perfect predictions (45-degree line)
plt.plot([min(np.exp(y_val)), max(np.exp(y_val))], [min(np.exp(y_val)), max(np.exp(y_val))], color='red', linestyle='--')

# Labels and title
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted Sale Price')
plt.grid(True)

# Show the plot
plt.show()

# Step 10: Make Predictions on Test Data
test_predictions = np.exp(best_model.predict(X_test_scaled))  # Convert back from log scale

# Step 11: Prepare and Save Submission File
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})
submission.to_csv("data/submission.csv", index=False)

print("Predictions saved successfully in 'data/submission.csv'")
