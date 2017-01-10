import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scipy.spatial import distance

def d(a, b):
    return distance.euclidean(a, b)

class KNearestNeighbors():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = d(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = d(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

clf = KNearestNeighbors()
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))