import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

C1 = np.array([[2, 12],
               [2.5, 12.5],
               [3, 12],
               [2.5, 11.5],
               [4, 12],
               [4.5, 12.5],
               [2, 11],
               [1.75, 10.75],
               [3, 11],
               [3.25, 11.25],
               [4, 11],
               [3.75, 10.75]])

C2 = np.array([[9, 2],
               [9.5, 2.5],
               [10, 2],
               [9.5, 1.5],
               [9, 3],
               [9.5, 3.5],
               [10, 3],
               [9.75, 2.75],
               [9, 4],
               [9.25, 4.25],
               [10, 4],
               [9.75, 3.75]])

C_12 = np.concatenate((C1, C2))
y_12 = np.zeros(len(C_12))

C3 = np.array([[5, 5],
               [5.5, 5.5],
               [6, 5],
               [5.5, 4.5],
               [5, 6],
               [5.5, 6.5],
               [6, 6],
               [5.75, 5.75],
               [5, 7],
               [5.25, 7.25],
               [6, 7],
               [5.75, 6.75]])

C4 = np.array([[7, 7],
               [7.5, 7.5],
               [7, 8],
               [6.5, 7.5],
               [8, 7],
               [8.5, 7.5],
               [8, 8],
               [7.75, 7.75],
               [9, 7],
               [9.25, 7.25],
               [9, 8],
               [8.75, 7.75]])


C_34 = np.concatenate((C3, C4))
y_34 = np.ones(len(C_34))

X = np.concatenate((C_12, C_34))
y = np.concatenate((y_12, y_34))

test_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

clf = MLPClassifier(solver='lbfgs', activation='tanh', 
                    hidden_layer_sizes=10, max_iter=200, random_state=1)

clf.fit(X_train, y_train)

test = np.array([[2.37, 7.45]])

pred = clf.predict(test)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

disp = DecisionBoundaryDisplay.from_estimator(clf, X_train, response_method="predict", alpha=0.3)

accuracy = accuracy_score(y_test, y_test_pred)
print('The accuracy of the model is: ', accuracy*100, '%')

disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, edgecolor='k')

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, edgecolor='k')
plt.scatter(test[:, 0], test[:, 1], c=pred, edgecolor='k')

plt.annotate('(2.37, 7.45)', (test[0, 0], test[0, 1]))

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.show()