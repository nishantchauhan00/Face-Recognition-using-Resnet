from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from data_loader import data

X_train, X_test, y_train, y_test, n_classes = data()
# X_train = preprocess_input(X_train)
# X_test = preprocess_input(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

img_height, img_width = X_train[0].shape[0], X_train[0].shape[1]
num_classes = n_classes

model = Sequential()

model.add(
    ResNet50(
        include_top=False,
        pooling="avg",
        weights="imagenet",
        input_shape=(img_height, img_width, 3),
    )
)
model.add(Dense(1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation="softmax"))

earlyStopping = EarlyStopping(
    monitor="val_loss", restore_best_weights=True, patience=10, verbose=0, mode="min"
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, callbacks=[earlyStopping])


preds = model.evaluate(X_test, y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
model.summary()

model.save("model.h5", overwrite=True)
