import random
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, walk
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

DATASET_DIR = "./dataset"
SPLITTING_CONST = 5

train_set, test_set = list(), list()

for _ , dirnames , _ in walk(DATASET_DIR):
    for sub in dirnames:
        dir_path = f"{DATASET_DIR}/{sub}"
        sub_set = list()

        for blob_name in listdir(dir_path):
            img = Image.open(path.join(dir_path , blob_name)).convert('L') 
            face = np.array(img).ravel()
            label = f"{sub}"
            sub_set.append([face, label])

        random.shuffle(sub_set)
        train_set.extend(sub_set[:SPLITTING_CONST])
        test_set.extend(sub_set[SPLITTING_CONST:])

train_set, test_set = np.array(train_set), np.array(test_set)

train_data, train_labels = np.hsplit(train_set, 2)
test_data, test_labels = np.hsplit(test_set, 2)
train_data = np.array([x[0] for x in train_data])
test_data = np.array([x[0] for x in test_data])
train_labels = np.array([x[0] for x in train_labels])
test_labels = np.array([x[0] for x in test_labels])

print(train_set[-1], [train_data[-1], train_labels[-1]])

train_mean = train_data.mean(axis=0)
center_matrix = train_data - train_mean
covariance_matrix = np.cov(center_matrix, rowvar=False, bias=True)
print("Convariance Matrix is here..")
eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
eigen_values = np.flip(eigen_values, axis=0)
eigen_vectors = np.flip(eigen_vectors, axis=1)

print(np.shape(covariance_matrix))

def choose_dimensionality(eigen_values, threshold):
    total_variance = np.sum(eigen_values)
    variance_fraction, num_dim, eigen_values_sum = 0, 1, 0
    while(variance_fraction < threshold):
        eigen_values_sum += eigen_values[num_dim-1]
        variance_fraction = eigen_values_sum / total_variance
        num_dim += 1
    return num_dim

alphas = [0.8, 0.85, 0.9, 0.95]
dims = [choose_dimensionality(eigen_values, alpha) for alpha in alphas]

projection_matrices = [eigen_vectors[:,0:int(dim)] for dim in dims]
print(np.shape(projection_matrices[0]))
reduced_train_data = [proj_mat.T @ center_matrix.T for proj_mat in projection_matrices]
print(reduced_train_data)

test_mean = test_data.mean(axis=0)
test_center_matrix = test_data - test_mean
reduced_test_data = [proj_mat.T @ test_center_matrix.T for proj_mat in projection_matrices]

def knn(train_data, train_labels, test_data, test_labels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    prediction = knn.predict(test_data)
    return metrics.accuracy_score(test_labels, prediction)

print([knn(reduced_train.T, train_labels, reduced_test.T, test_labels, 1) for reduced_train, reduced_test in zip(reduced_train_data, reduced_test_data)])
