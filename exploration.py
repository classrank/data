import numpy as np
import sklearn
import sklearn.cross_validation
import sklearn.tree
import sklearn.svm

with open('ratings.csv') as f:
    data = [line[:-1].replace("\"", "").split(",") for line in f.readlines()[1:]]

# remove the timestamps
for item in data:
    item.pop(0)

# split the threads out
threads = ['Devices', 'Info & Internetworks', 'Intelligence', 'Media', 'Modeling & Simulation', 'People', 'Systems & Architecture', 'Theory', 'I graduated long ago (None)']
for item in data:
    item[0] = [1 if (thread in set(item[0].split(";"))) else 0 for thread in threads]

#set up nans
for row in data:
    for i in range(1, len(row)):
        if row[i] == "Very Easy":
            row[i] = 1
        elif row[i] == 'Pretty Easy':
            row[i] = 2
        elif row[i] == "Average":
            row[i] = 3
        elif row[i] == "Difficult":
            row[i] = 4
        elif row[i] == "Nearly Impossible":
            row[i] = 5
        else:
            row[i] = "NaN"

x = np.array([[0 if (i== "NaN") else i for i in row[1:]] for row in data])
y = np.array([row[0][]3 for row in data])

learner = sklearn.tree.DecisionTreeClassifier()

for train, test in sklearn.cross_validation.KFold(len(x), n_folds=10):
    real = learner.fit(x[train], y[train]).score(x[test], y[test])
    base = 1 - sum(y[test])*1.0/len(y[test])
    print(base)
    print(real)
    print(1.0*(real - base) / (1- base) / 2 + .5 if real != 1 else real)
    print("------")