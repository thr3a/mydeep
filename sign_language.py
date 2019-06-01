import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0",
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

batch_size = 32
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 100, 100

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(img_rows, img_cols, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# the data, split between train and test sets
train_dir = '/works/data/hand_sign/train'
validation_dir = '/works/data/hand_sign/test'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

log_dir="works/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_grads=True,
    write_graph=False)

os.makedirs('/works/models', exist_ok=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('/works/models', 'model_{epoch:02d}_{val_loss:.2f}.h5'),
    monitor='val_loss',
    verbose=1)

train_step_size = train_generator.n // train_generator.batch_size
validation_step_size = validation_generator.n // validation_generator.batch_size
print(train_step_size)
print(validation_step_size)
model.fit_generator(
      train_generator,
      steps_per_epoch=train_step_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=validation_step_size,
      verbose=1,
      callbacks=[tensorboard_callback, model_checkpoint]
      )

# model.save('sign_1.h5')
