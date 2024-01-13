import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

model_name = "weatherclassifier"

def train():
    # configure nvidia gpu
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    data_dir = "data"

    
    # img = cv2.imread(os.path.join(data_dir, "rain", "1011.jpg"))

    # class 0 -> rain
    # class 1 -> snow
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    
    data_iter = data.as_numpy_iterator()
    batch = data_iter.next()

    # normalize images to have color values between [0, 1]
    scaled = data.map(lambda x, y: (x / 255, y))
    batch = scaled.as_numpy_iterator().next()

    # set the sizes of train, validation, and test data    
    size_train = int(len(scaled) * 0.7)
    size_valid = int(len(scaled) * 0.2) + 1
    size_test = int(len(scaled) * 0.1)

    if size_train + size_valid + size_test != len(scaled):
        print("Train, validation, and test batches does not add up to total data batches")

    # set train, validation, and test data
    data_train = scaled.take(size_train)
    data_valid = scaled.skip(size_train).take(size_valid)
    data_test = scaled.skip(size_train + size_valid).take(size_test)

    if len(data_train) + len(data_valid) + len(data_test) != len(scaled):
        print("Train, validation, and test data lengths does not add up to total data batches")

    # configure the model
    model = Sequential()
    
    height, width, channels = scaled.element_spec[0].shape[1:]
    
    model.add(Conv2D(16, (3, 3), 1, activation="relu", input_shape=(height, width, channels)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3, 3), 1, activation="relu"))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid")) # determines a value between 0 and 1
    
    model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])
    
    log_dir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # train the model
    history = model.fit(data_train, epochs=25, validation_data=data_valid, callbacks=tensorboard_callback)
    
    #  plot loss graph
    fig_performance = plt.figure()
    plt.plot(history.history["loss"], color="pink", label="loss")
    plt.plot(history.history["val_loss"], color="purple", label="validation_loss")
    fig_performance.suptitle("Loss", fontsize=20)
    plt.legend()
    plt.show()

    # plot accuracy graph    
    fig_performance = plt.figure()
    plt.plot(history.history["accuracy"], color="teal", label="accuracy")
    plt.plot(history.history["val_accuracy"], color="purple", label="val_accuracy")
    fig_performance.suptitle("Accuracy", fontsize=20)
    plt.legend()
    plt.show()

    # evaluate the model
    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()
    
    for batch in data_test.as_numpy_iterator():
        x, y = batch
        yhat = model.predict(x)
        precision.update_state(y, yhat)
        recall.update_state(y, yhat)
        accuracy.update_state(y, yhat)
    
    print(f"precision: {precision.result().numpy()}\nrecall: {recall.result().numpy()}\naccuracy: {accuracy.result().numpy()}\n")
    
    # save the model
    model.save(os.path.join("model", "weatherclass1.h5"))


# %%

# %% [markdown]
# Test

# %%
test_dir = "test_data"

# %%
img = cv2.imread(os.path.join(test_dir, "snow3.jpg"))
img.shape
plt.imshow(img)

# %%
resized = tf.image.resize(img, (256, 256))
plt.imshow(resized.numpy().astype(int))

# %%
yhat = model.predict(np.expand_dims(resized/255, 0))
yhat

# %%
if yhat < 0.5:
    print("Image is rain")
else:
    print("Image is snow")

# %% [markdown]
# Save Model

# %%


# %%


# %%
saved_model = load_model(os.path.join("model", "weatherclass.h5"))

# %%
yhat = model.predict(np.expand_dims(resized/255, 0))
yhat

# %%
if yhat < 0.5:
    print("Image is rain")
else:
    print("Image is snow")

# %%


# %%



def save_model():
    