import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score

"""
для датасету mnist
1) провести нормалізацію в 0-1
2) натренувати модель логістичної регресії для класів y>=5, y<5
3) використати PCA для зменшення розмірності, і на новому датасеті для тих самих класів наттренувати логістичну регресію
4) порівняти точність моделей 2 і 3
"""

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Training Data: {}'.format(x_train.shape))
print('Training Labels: {}'.format(y_train.shape))
print("labels:", y_train)

# creating 1D array and normalization
x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255
print(x_train.shape, x_test.shape)

# creating binary labels for class y>=5(1) and y<5(0)
y_train_b = (y_train >= 5).astype(int)
y_test_b = (y_test >= 5).astype(int)

# crating logistic regression
model_logistic_regression = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train_b)

# predict set test
predict = model_logistic_regression.predict(x_test)

# accuracy
accuracy = accuracy_score(y_test_b, predict)

# PCA
pca = PCA(330)
new_x_train = pca.fit_transform(x_train)
new_x_test = pca.transform(x_test)

print('\nPCA saved:', pca.explained_variance_ratio_.sum(), ' of dataset')

# # calculating how many features need for saving 0.99% dataset(330 feature)
# cum_sum = np.cumsum(pca.explained_variance_ratio_) >= 0.99
# print('\n', cum_sum)
#
# count_cum_sum = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99)
# print('\n', count_cum_sum)

# creating model with new dataset
new_model = LogisticRegression(random_state=0, max_iter=1000).fit(new_x_train, y_train_b)

# predict
new_predict = new_model.predict(new_x_test)

# accuracy
new_accuracy = accuracy_score(y_test_b, new_predict)
print(f'Accuracy of logistic model: {accuracy}')
print(f'Accuracy of logistic model using PCA: {new_accuracy}')

# comparison
if new_accuracy > accuracy:
    print('Accuracy of logistic model using PCA better')
elif accuracy > new_accuracy:
    print('Accuracy of logistic model better')
else:
    print('Result equals')
