import pandas as pd
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('ames.csv')
print(df.head())

# Show missing values per column
missing_counts = df.isnull().sum()
missing_columns = missing_counts[missing_counts > 0].sort_values(ascending=False)
print(missing_columns)

# working on missing values
df.drop(columns=['Pool.QC', 'Misc.Feature', 'Alley', 'Fence'], inplace=True)
df['Mas.Vnr.Type'] = df['Mas.Vnr.Type'].fillna(df['Mas.Vnr.Type'].mode()[0])
df['Fireplace.Qu'] = df['Fireplace.Qu'].fillna('None')
df['Lot.Frontage'] = df['Lot.Frontage'].fillna(df['Lot.Frontage'].median())
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Fill basement-related categorical columns with 'None'
bsmt_cols = ['Bsmt.Exposure', 'BsmtFin.Type.2', 'Bsmt.Cond', 'Bsmt.Qual', 'BsmtFin.Type.1']
for col in bsmt_cols:
    df[col] = df[col].fillna('None')

# Fill garage-related categorical columns with 'None'
garage_cols = ['Garage.Qual', 'Garage.Cond', 'Garage.Finish', 'Garage.Type']
for col in garage_cols:
    df[col] = df[col].fillna('None')

# Fill numeric columns with 0 or median, depending on context
df['Garage.Yr.Blt'] = df['Garage.Yr.Blt'].fillna(0)
df['Mas.Vnr.Area'] = df['Mas.Vnr.Area'].fillna(0)

numeric_fill = [
    'Bsmt.Full.Bath', 'Bsmt.Half.Bath', 'BsmtFin.SF.1', 'BsmtFin.SF.2',
    'Total.Bsmt.SF', 'Bsmt.Unf.SF', 'Garage.Area', 'Garage.Cars'
]
for col in numeric_fill:
    df[col] = df[col].fillna(df[col].median())
# Check total number of missing values remaining
total_nulls = df.isnull().sum().sum()
print(f'Total missing values remaining: {total_nulls}')

# Function to calculate Z-Score and detect outliers
def detect_outliers_zscore(df, column, threshold=3):
    # Calculate Z-Score for the specified column
    z_scores = zscore(df[column])
    # Find outliers where absolute Z-Score is greater than threshold
    outliers = (np.abs(z_scores) > threshold).sum()
    return outliers

# Create a dictionary to store outliers count for each column
outliers_counts = {}

# Loop through each numeric column in the dataframe
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    outliers_counts[column] = detect_outliers_zscore(df, column)

# Print out the number of outliers for each column
for column, count in outliers_counts.items():
    print(f'{column}: {count} outliers')

# Remove rows with any numeric value having Z-score > 3
numeric_df = df.select_dtypes(include=['float64', 'int64'])
z_scores = np.abs(zscore(numeric_df))
df = df[(z_scores < 3).all(axis=1)]
print(f'Dataset shape after removing outliers: {df.shape}')

# Feature creation
print(df.columns.tolist())
df['Total.Bathrooms'] = df['Full.Bath'] + (0.5 * df['Half.Bath']) + df['Bsmt.Full.Bath'] + (0.5 * df['Bsmt.Half.Bath'])
df['Total.SF'] = df['Total.Bsmt.SF'] + df['X1st.Flr.SF'] + df['X2nd.Flr.SF']
df['House.Age'] = df['Yr.Sold'] - df['Year.Built']
df['Is.Remodeled'] = (df['Year.Remod.Add'] != df['Year.Built']).astype(int)
df['Total.Porch.SF'] = df['Open.Porch.SF'] + df['Enclosed.Porch'] + df['X3Ssn.Porch'] + df['Screen.Porch']

# One-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Data shape after encoding:", df.shape)

# Compute the correlation matrix for numeric features
corr_matrix = df.corr(numeric_only=True)

# Sort correlations with SalePrice in descending order
saleprice_corr = corr_matrix['SalePrice'].sort_values(ascending=False)
print("Top correlated features with SalePrice:")
print(saleprice_corr.head(10))  # Top 10 most positively correlated features
print("\nLeast correlated features with SalePrice:")
print(saleprice_corr.tail(5))   # Least correlated features

# Plot the heatmap for top correlated features
top_features = saleprice_corr.index[:10]
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Top Features")
plt.tight_layout()
plt.show()

# Correlation matrix for numeric columns
correlation_matrix = df.corr(numeric_only=True)

# Get absolute correlation with SalePrice and sort
correlation_with_target = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)

# Select features with correlation > 0.5 (you can adjust threshold)
selected_features = correlation_with_target[correlation_with_target > 0.5].index.tolist()

# Remove 'SalePrice' from the list itself if you only want features
selected_features.remove('SalePrice')
print("Selected features (correlation > 0.5):")
print(selected_features)

# Create new DataFrame with selected features + target
df_selected = df[selected_features + ['SalePrice']]

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split data into training and testing sets
X = df_selected.drop(columns=['SalePrice'])
y = df_selected['SalePrice']

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print("\n"*1)
# Step 5: Visualize the predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual SalePrice")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.tight_layout()
plt.show()

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Gradient Boost Regressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Define the model
gb_model = GradientBoostingRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'max_depth': [3, 5, 7],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required to be at a leaf node
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Evaluate the best model
best_gb_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best_gb = best_gb_model.predict(X_test)

# Evaluate the model
mae_best_gb = mean_absolute_error(y_test, y_pred_best_gb)
mse_best_gb = mean_squared_error(y_test, y_pred_best_gb)
r2_best_gb = r2_score(y_test, y_pred_best_gb)

print("Best Gradient Boosting Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_best_gb}")
print(f"Mean Squared Error (MSE): {mse_best_gb}")
print(f"R-squared (R²): {r2_best_gb}")
