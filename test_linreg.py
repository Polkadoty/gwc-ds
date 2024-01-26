import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import cudf
from cuml.linear_model import LinearRegression

class LinearRegressionTest(unittest.TestCase):
    def setUp(self):
        # Load the data from the CSV file
        self.data = cudf.read_csv('melb_data.csv')

        # Separate the features (X) and the target variable (y)
        self.X = self.data.drop('Price', axis=1)
        self.y = self.data['Price']

        # One-hot encode categorical variables
        self.X_encoded = cudf.get_dummies(self.X)

        # Impute missing values
        self.imputer = cudf.SimpleImputer()
        self.X_imputed = self.imputer.fit_transform(self.X_encoded)

        # Create an instance of the Linear Regression model
        self.model = LinearRegression()

        # Fit the model to the encoded data
        self.model.fit(self.X_imputed, self.y)

    def test_accuracy(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_imputed, self.y, test_size=0.2, random_state=42)

        # Predict the target variable for the test set
        y_pred = self.model.predict(X_test)

        # Calculate the mean squared error as a measure of accuracy
        mse = mean_squared_error(y_test, y_pred)

        # Assert that the mean squared error is below a certain threshold
        self.assertLess(mse, 1e9)  # Adjust the threshold as needed

if __name__ == '__main__':
    unittest.main()