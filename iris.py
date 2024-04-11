import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Load dataset
iris = pd.read_csv('Iris.csv')

#print first 5 
print(iris.head())

#sepal width greater than 4 cm
print(iris[iris['SepalWidthCm'] > 4])

#petal width greather than 1 cm
print(iris[iris['PetalWidthCm'] > 1])

#petal width greather than 2 cm
print(iris[iris['PetalWidthCm'] > 2])

# Basic scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris, x='SepalLengthCm', y='PetalLengthCm', hue ='Species')
plt.title('Sepal Length vs Petal Length')
plt.show()


# Model 1
# Define x and y
x = iris[['SepalWidthCm']]
y = iris['SepalLengthCm']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

# Predictions
y_pred = lr.predict(x_test)

# Show first five predictions
print("Actual:", y_test.head())
print("Predicted:", y_pred[:5])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squarred Error:", mse)


# Model 2
# Define x and y
x = iris[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['SepalLengthCm']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)  # 70% training, 30% testing

from sklearn.linear_model import LinearRegression

# Initialize the model
lr = LinearRegression()

# Fit the model on the training data
lr.fit(x_train, y_train)

# Predictions
y_pred = lr.predict(x_test)

# Display the first five actual and predicted values
print("Actual:", y_test.head())
print("Predicted:", y_pred[:5])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
