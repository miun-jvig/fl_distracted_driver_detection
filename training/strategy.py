from training.samplingdata import BalancedDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from training.utils import mix_up, train_val_split, preprocess_labels, compute_class_weight


def input_processing(augmentation=True):
    if augmentation:
        train_datagen = ImageDataGenerator(
            zoom_range=[0.5, 2.3],
            brightness_range=[0.5, 2.3],
            rotation_range=30,
            horizontal_flip=True,
            height_shift_range=0.3,
            width_shift_range=0.3,
            fill_mode='nearest',
            shear_range=0.2,
        )
    else:
        train_datagen = ImageDataGenerator()
    validation_datagen = ImageDataGenerator()
    return train_datagen, validation_datagen


def batch_image_generator(images, labels, batch_size, data_augmentation, data_subpart=False):
    sample_count = len(labels)
    while True:
        for index in range(sample_count//batch_size):
            start = index * batch_size
            end = (index + 1) * batch_size
            ys = labels[start: end]
            if data_subpart:
                full, faces, hands = images[0], images[1], images[2]
                generator_full = data_augmentation.flow(full[start: end], ys, batch_size=batch_size)
                generator_faces = data_augmentation.flow(faces[start: end], ys, batch_size=batch_size)
                generator_hands = data_augmentation.flow(hands[start: end], ys, batch_size=batch_size)
                x_faces = generator_faces.next()
                x_full = generator_full.next()
                x_hands = generator_hands.next()
                yield [x_faces[0], x_full[0], x_hands[0]], ys
            else:
                xs = images[start: end]
                generator = data_augmentation.flow(xs, ys, batch_size=batch_size).next()
                yield generator[0], generator[1]


def data_generator(x_train, y_train, x_vali, y_vali, gen_type='vanilla', batch_size=32, train_gen=None, vali_gen=None, input_shape=(128,128,3)):
    if 'un' in gen_type.lower():
        training_generator = BalancedDataGenerator(x_train, y_train, train_gen,
                                                       batch_size=batch_size, undersampling=True)
        validation_generator = BalancedDataGenerator(x_vali, y_vali, vali_gen,
                                                         batch_size=batch_size, undersampling=True)
        return training_generator, validation_generator

    if 'ov' in gen_type.lower():
        training_generator = BalancedDataGenerator(x_train, y_train, train_gen,
                                                       batch_size=batch_size, undersampling=False)
        validation_generator = BalancedDataGenerator(x_vali, y_vali, vali_gen,
                                                         batch_size=batch_size, undersampling=False)
        #steps_per_epoch = balanced_gen_train.steps_per_epoch
        return training_generator, validation_generator

    if 'mixup' in gen_type.lower():
        AUTO = tf.data.AUTOTUNE
        train_ds_one = tf.data.Dataset.from_generator(
            lambda: batch_image_generator(x_train, y_train, batch_size, train_gen),
            output_types=(tf.float32, tf.float32),
            output_shapes=([batch_size, input_shape[0], input_shape[1], input_shape[2]], [batch_size, 3])
        )

        p = np.random.permutation(len(y_train))

        train_ds_two = tf.data.Dataset.from_generator(
            lambda: batch_image_generator(x_train[p], y_train[p], batch_size, train_gen),
            output_types=(tf.float32, tf.float32),
            output_shapes=([batch_size, input_shape[0], input_shape[1], input_shape[2]], [batch_size, 3])
        )

        # Because we will be mixing up the images and their corresponding labels, we will be
        # combining two shuffled datasets from the same training data.
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        # First create the new dataset using our `mix_up` utility
        training_generator = train_ds.map(
            lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.1), num_parallel_calls=AUTO
        )

        validation_generator = tf.data.Dataset.from_generator(
            lambda: batch_image_generator(x_train, y_train, batch_size, train_gen),
            output_types=(tf.float32, tf.float32),
            output_shapes=([batch_size, input_shape[0], input_shape[1], input_shape[2]], [batch_size, 3])
        )
        return training_generator, validation_generator

    # vanilla version of the batch generator
    training_generator = batch_image_generator(x_train, y_train, batch_size, data_augmentation=train_gen)
    validation_generator = batch_image_generator(x_vali, y_vali, batch_size, data_augmentation=vali_gen)
    return training_generator, validation_generator


def fitting(training_generator, validation_generator, model, cid, warmstart,
            nb_steps_training, nb_steps_val, modelname, nb_epoch=10, class_weight=None):
    path = "./logs/" + modelname + "/client-" + cid + "/"
    checkpoint_path = path + "/cpft-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')
    csv_eval_path = path + "/training_startingfrom" + warmstart + ".log"
    csv_callback = tf.keras.callbacks.CSVLogger(csv_eval_path)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(training_generator,
                        steps_per_epoch=nb_steps_training,
                        epochs=nb_epoch,
                        verbose=1,
                        class_weight=class_weight,
                        validation_data=validation_generator,
                        validation_steps=nb_steps_val,
                        initial_epoch=int(warmstart),
                        callbacks=[cp_callback, csv_callback, early_stopping])
    return model, history


def training_model(x, y, model, modelname, nb_epoch=10, cid=0, warmstart='0', batch_size=32, generator_type='vanilla',
                   augmentation=True, class_weight=False, vali_ratio=0.2, input_shape=(128, 128, 3)):

    # data augmentation or not?
    train_datagen, validation_datagen = input_processing(augmentation)
    cw = None
    if class_weight:
        cw = compute_class_weight(y)

    y = preprocess_labels(y, len(np.unique(y)))

    # split the set into train and validation
    x_train, y_train, x_vali, y_vali = train_val_split(x, y, vali_ratio)
    # how the batch is created, using: mixup, oversampling, undersampling, or vanilla version
    training_generator, validation_generator = data_generator(x_train, y_train, x_vali, y_vali, generator_type,
                                                              batch_size, train_datagen, validation_datagen,
                                                              input_shape)

    if 'un' in generator_type.lower() or 'ov' in generator_type.lower():
        nb_steps_training = training_generator.steps_per_epoch
        nb_steps_val = validation_generator.steps_per_epoch
    else:
        nb_steps_training = len(y_train) // batch_size
        nb_steps_val = len(y_vali) // batch_size
    return fitting(training_generator, validation_generator, model, cid, warmstart,
                   nb_steps_training, nb_steps_val, modelname, nb_epoch=nb_epoch, class_weight=cw)
