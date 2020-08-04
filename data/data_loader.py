from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2.cv2 as cv2
# from sklearn.preprocessing import OneHotEncoder

def data():
    lfw_people = fetch_lfw_people(
        min_faces_per_person=3,
        color=True,
        slice_=(slice(0, 250, None), slice(0, 250, None)),
    )

    # The original images are 250 x 250 pixels,
    # but the default slice and resize arguments reduce them to 62 x 47 pixels.

    # introspect the images arrays to find the shapes
    n_samples, h, w, pixel = lfw_people.images.shape

    X = lfw_people.images

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    # print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    print("Image size: %dx%d" % (h, w))

    # encoder = OneHotEncoder()
    # y = encoder.fit_transform(y.reshape(-1,1))

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test, n_classes
