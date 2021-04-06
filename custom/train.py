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
from model import build_model


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
    model = build_model(num_classes=num_classes)

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

    # ----- SAVE ----- #
    # save model
    model.save(SAVE_DIR)

    # ----- DEPLOY ----- #
    # convert model to TF Lite
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    tflite_model = converter.convert()

    with open(os.path.join(SAVE_DIR, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
