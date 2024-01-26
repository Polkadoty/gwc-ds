import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import cudf
from cuml.linear_model import LinearRegression
import time

class LinearRegressionTest(unittest.TestCase):
    def setUp(self):
        start_time = time.time()  # Start the timer

        # Load the data from the CSV file
        self.data = cudf.read_csv('melb_data.csv')

        # Separate the features (X) and the target variable (y)
        self.X = self.data.drop('Price', axis=1)
        self.y = self.data['Price']

        # One-hot encode categorical variables
        self.X_encoded = cudf.get_dummies(self.X)

        # Impute missing values with mean
        self.X_imputed = self.X_encoded.fillna(self.X_encoded.mean())

        # Create an instance of the Linear Regression model
        self.model = LinearRegression()

        # Fit the model to the encoded data
        self.model.fit(self.X_imputed, self.y)

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")


    def test_accuracy(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_imputed, self.y, test_size=0.2, random_state=42)

        # Predict the target variable for the training set
        y_pred_train = self.model.predict(X_train)

        # Calculate the mean squared error for the training set as a measure of accuracy
        mse_train = mean_squared_error(y_train, y_pred_train)

        # Predict the target variable for the testing set
        y_pred_test = self.model.predict(X_test)

        # Calculate the mean squared error for the testing set as a measure of accuracy
        mse_test = mean_squared_error(y_test, y_pred_test)

        # Assert that the mean squared error for both training and testing sets are below a certain threshold
        self.assertLess(mse_train, 1e9)  # Adjust the threshold as needed
        self.assertLess(mse_test, 1e9)  # Adjust the threshold as needed

if __name__ == '__main__':
    unittest.main()