import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('score.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1 )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Hours Studied vs. Score Acheived (Test set)")
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()

print(regressor.predict([[8]]))
print(regressor.coef_)
print(regressor.intercept_)

from sklearn import metrics
print("Mean Absolute Error =", metrics.mean_absolute_error(y_test, y_pred))
print("RMSE =", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score =", metrics.r2_score(y_test, y_pred))
