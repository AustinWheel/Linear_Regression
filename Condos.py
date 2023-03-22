import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("condos.csv")
X = dataset.iloc[:, 2:5].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.29, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(regressor.coef_)
print(regressor.intercept_)
print("Prediction output: ", regressor.predict([[9, 1, 2, 2]]))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

x1, x2, x3 = [], [], []
for i in range(len(X_train[0])):
    for j in range(len(X_train)):
        if i == 0:
            x1.append(X_train[j][i])
        if i == 1:
            x2.append(X_train[j][i])
        if i == 2:
            x3.append(X_train[j][i])

ax.scatter(x1, x2, x3)
ax.set_xlabel("X1 Label")
ax.set_ylabel("X2 Label")
ax.set_zlabel("X3 Label")

plt.show()


from sklearn import metrics
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 score =", metrics.r2_score(y_test, y_pred))