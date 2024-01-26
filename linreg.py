import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
# Load the data from the CSV file
data = pd.read_csv('melb_data.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Price', axis=1)
y = data['Price']
# One-hot encode categorical variables
X_encoded = pd.get_dummies(X)

# Impute missing values
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X_encoded)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the encoded data
model.fit(X_imputed, y)

