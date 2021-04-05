"""
Michael Patel
April 2021

Project description:
    Use TensorFlow to create a Simpsons character classifier that runs on an Android device

File description:
    For model preprocessing and training

"""
################################################################################
# Imports
from packages import *


################################################################################
# Main
if __name__ == "__main__":
    # ----- SETUP ----- #
    # create a save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # ----- ETL ----- #
    # get labels
    character_directories = os.listdir(DATA_DIR)
    num_classes = len(character_directories)
    #print(f'Number of classes: {num_classes}')

    # create a text file with labels
    labels_filepath = os.path.join(SAVE_DIR, "labels.txt")
    if not os.path.exists(labels_filepath):
        with open(labels_filepath, "w") as f:
            for d in character_directories:
                f.write(d + "\n")

    # image generators
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        #rotation_range=30,
        #horizontal_flip=True,
        #vertical_flip=True,
        #width_shift_range=0.3,
        #height_shift_range=0.3,
        #brightness_range=[0.3, 1.3],
        validation_split=VALIDATION_SPLIT
    )

    train_generator = datagen.flow_from_directory(
        directory=DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        directory=DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        subset="validation"
    )

    # ----- MODEL ----- #
    # build model
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        weights="imagenet",
        include_top=False
    )
    mobilenet.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    x = inputs
    x = mobilenet(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    outputs = x

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    # compile model
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )

    model.summary()

    # ----- TRAIN ----- #
    history = model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )

    # training plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 3.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(os.path.join(SAVE_DIR, "plots"))

    # ----- FINE TUNE ---- #
    print(f'\n\nFINE TUNE\n\n')
    
    # re-build model
    # unfreeze base
    mobilenet.trainable = True
    for layer in mobilenet.layers[:NUM_LAYERS_FREEZE]:
        layer.trainable = False  # freeze early layers

    # re-compile model
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_FINE_TUNING),
        metrics=["accuracy"]
    )

    model.summary()

    # continue training
    history = model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS_FINE_TUNING,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )

    # fine-tuning plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 3.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(os.path.join(SAVE_DIR, "plots_finetune"))

    # ----- SAVE ----- #
    # save model

    # ----- DEPLOY ----- #
    # convert model to TF Lite
