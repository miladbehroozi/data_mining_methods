import pandas as pd
import numpy as np
from scipy.stats import zscore

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
