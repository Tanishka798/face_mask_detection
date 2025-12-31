import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

baseModel= MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten()(headModel)
headModel=Dense(64,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation="softmax")(headModel)

model=Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers :
    layer.trainable=False

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

trainGen=datagen.flow_from_directory(
    "data",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

valGen=datagen.flow_from_directory(
    "data",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

H=model.fit(
    trainGen,
    validation_data=valGen,
    epochs=10,
    callbacks=[early_stop]
)

model.save("mask_detector.keras")

plt.plot(H.history["accuracy"],label="train_acc")
plt.plot(H.history["val_accuracy"],label="val_acc")
plt.legend()
plt.title("Training Accuracy")
plt.show()