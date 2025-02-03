# Import necessary libraries for data manipulation, visualization, and modeling
import pandas as pd  # For working with dataframes
import numpy as np  # For numerical operations like logarithms and array manipulations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For advanced data visualization (heatmaps, scatter plots, etc.)
from sklearn.linear_model import LinearRegression  # For creating a linear regression model
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.ensemble import RandomForestRegressor  # For creating a random forest regression model

# Load the dataset
data = pd.read_csv("housing.csv")  # Read the housing data from a CSV file into a pandas DataFrame
print(data.info())  # Print the summary information about the dataset (data types, number of missing values, etc.)

# Handle missing values
data.dropna(inplace=True)  # Remove rows with missing values (if any)
print(data.info())  # Print summary info again to ensure there are no missing values now

# Prepare the features (X) and target variable (y)
X = data.drop(['median_house_value'], axis=1)  # Drop the target column ('median_house_value') to create the feature set
print(X)  # Print the feature set (X) to check its structure

y = data['median_house_value']  # Extract the target variable 'median_house_value'
print(y)  # Print the target variable to verify

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Randomly split the dataset
X_train.reset_index(drop=True, inplace=True)  # Reset the index of the training set (for convenience)
y_train.reset_index(drop=True, inplace=True)  # Reset the index of the target variable (for convenience)

# Visualize the distribution of features in the training data
train_data = X_train.join(y_train)  # Join the feature set and target variable to create a full training dataset
train_data.hist(figsize=(15, 8))  # Plot histograms for all numerical columns in the training data
plt.show()  # Display the histograms

# One-hot encode categorical variable 'ocean_proximity' and drop the original column
train_data = train_data.join(pd.get_dummies(train_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)
# This converts the 'ocean_proximity' categorical feature into numerical columns (0/1), representing the different categories

# Check the correlation matrix of the training data to see how features are related to each other and to the target variable
corr_matrix = train_data.corr()  # Calculate correlation matrix for all features
print(corr_matrix)  # Print the correlation matrix

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(15, 8))  # Set the size of the heatmap
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")  # Plot the heatmap with annotations (values inside the squares)
plt.show()  # Display the heatmap

# Apply logarithmic transformation to certain features to reduce skewness and improve model performance
train_data = X_train.join(y_train)
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)  # Log transform 'total_rooms'
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)  # Log transform 'total_bedrooms'
train_data['population'] = np.log(train_data['population'] + 1)  # Log transform 'population'
train_data['households'] = np.log(train_data['households'] + 1)  # Log transform 'households'

# Visualize the distributions of transformed features
train_data.hist()  # Plot histograms of the transformed data
plt.show()  # Display the histograms

# Visualize the relationship between latitude and longitude, colored by median house value
plt.figure(figsize=(15, 8))  # Set the size of the scatter plot
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette='coolwarm')
plt.show()  # Display the scatter plot

# Create new features that might have predictive value
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']  # Bedroom-to-room ratio
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']  # Room-to-household ratio

# One-hot encode 'ocean_proximity' again after feature transformations
train_data = train_data.join(pd.get_dummies(train_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)

# Check the correlation matrix after the feature engineering
corr_matrix = train_data.corr()  # Recalculate the correlation matrix
print(corr_matrix)  # Print the updated correlation matrix

# Visualize the updated correlation matrix with a heatmap
plt.figure(figsize=(15, 8))  # Set the size of the heatmap
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")  # Plot the heatmap with annotations
plt.show()  # Display the heatmap

# Prepare the training data for model fitting
X_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']  # Separate features and target variable
reg = LinearRegression()  # Initialize a linear regression model
reg.fit(X_train, y_train)  # Fit the linear regression model to the training data

# Prepare and process the test data in the same way as the training data
test_data = X_test.join(y_test)  # Join test features and target variable
test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)  # Log transform test features
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

# One-hot encode the categorical feature 'ocean_proximity' in the test data
test_data = test_data.join(pd.get_dummies(test_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)

# Apply feature engineering transformations to the test data
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

# Prepare test data features and target variable
X_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
X_test = X_test[X_train.columns]  # Ensure that the test data has the same columns as the training data

# Evaluate the performance of the linear regression model on the test data
a = reg.score(X_test, y_test)  # Calculate the R^2 score (coefficient of determination) of the linear regression model
print(a)  # Print the R^2 score

# Train a random forest regression model on the training data
forest = RandomForestRegressor()  # Initialize a random forest regressor model
forest.fit(X_train, y_train)  # Fit the random forest model to the training data

# Evaluate the performance of the random forest model on the test data
a = forest.score(X_test, y_test)  # Calculate the R^2 score of the random forest model
print(a)  # Print the R^2 score
