import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.1

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate polynomial regression models
def fit_poly_regression(X_train, y_train, X_test, y_test, degree):
    # Polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Prediction on training and testing set
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Calculate mean squared error for bias-variance plot
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    return model, train_error, test_error, X_train_poly, X_test_poly

# Set the degree of polynomials to be tried
degrees = np.arange(1, 30)
train_errors = []
test_errors = []

# Fit models with different polynomial degrees
plt.figure(figsize=(12, 8))
plt.scatter(X, y, label='Data', color='black', s=30)

for degree in degrees:
    model, train_error, test_error, X_train_poly, X_test_poly = fit_poly_regression(X_train, y_train, X_test, y_test, degree)
    
    # Plot the fitted curve
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    X_plot_poly = PolynomialFeatures(degree=degree).fit_transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, label=f'Degree {degree}')

    train_errors.append(train_error)
    test_errors.append(test_error)

plt.title('Polynomial Regression with Bias-Variance Tradeoff')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Plot bias-variance tradeoff
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Training Error', marker='o')
plt.plot(degrees, test_errors, label='Testing Error', marker='o')
plt.title('Bias-Variance Tradeoff with Polynomial Regression')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
