import cv2 as cv
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Extract HOG features from the training set
hog_features_train = [hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1)) for img in x_train]

# Split the data into training and testing sets for SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(hog_features_train, y_train, test_size=0.2, random_state=42)

# SVM model
svm_model = svm.SVC()
svm_model.fit(X_train_svm, y_train_svm)

# Evaluate SVM accuracy
svm_predictions = svm_model.predict([hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1)) for img in x_test])
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f'SVM Accuracy: {svm_accuracy}')

# Extract HOG features from the test image
for x in range(1, 10):
    # Load image
    img = cv.imread(f'{x}.png')[:, :, 0]  # Convert to grayscale
    np_image = np.array(img)
    
    hog_feature_test, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)

    # SVM prediction
    svm_prediction = svm_model.predict([hog_feature_test])
    print(f'SVM Prediction for Image {x}: {svm_prediction[0]}')

    # Plot original image and HOG features
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(np_image, cmap=plt.cm.gray)
    ax[0].set_title('Original Image')

    ax[1].imshow(hog_image, cmap=plt.cm.gray)
    ax[1].set_title('HOG Features')

    plt.show()
