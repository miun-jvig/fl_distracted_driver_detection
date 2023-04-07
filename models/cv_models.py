import tensorflow as tf
from keras.models import Sequential, Model
from keras.applications import EfficientNetB0, Xception, EfficientNetB3, MobileNetV2, VGG16
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

def create_vgg16(input_shape, classes = 3, fclayers=[2048, 1024], trainable=False, init='imagenet'):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    # Remove fully connected layer and replace
    vgg16_model = VGG16(include_top=False, weights=init, input_shape=input_shape, classes=classes)

    for layer in vgg16_model.layers:
        layer.trainable = trainable

    x = tf.keras.layers.Flatten(name="flatten")(vgg16_model.output)

    for fc in fclayers:
        x = Dense(fc, activation="relu")(x)
        x = Dropout(0.3)(x)

    output = Dense(classes, activation="softmax")(x)
    model = Model(vgg16_model.input, output)
    return model


def create_xception(input_shape, classes=3, fclayers=[2048, 1024], trainable=False, init='imagenet'):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    # Remove fully connected layer and replace
    xception_model = Xception(include_top=False, weights=init, input_shape=input_shape, classes=classes)
    for layer in xception_model.layers:
        layer.trainable = trainable

    x = tf.keras.layers.GlobalAveragePooling2D()(xception_model.output)

    for fc in fclayers:
        x = Dense(fc, activation="relu")(x)
        x = Dropout(0.3)(x)

    output = Dense(classes, activation="softmax")(x)
    model = Model(xception_model.input, output)
    return model



def create_efficientB3(input_shape, classes=3, fclayers=[2048, 1024], trainable=False, init='imagenet'):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    # Remove fully connected layer and replace
    efficient_model = EfficientNetB3(include_top=False, weights=init, input_shape=input_shape, classes=classes)
    for layer in efficient_model.layers:
        layer.trainable = trainable

    x = tf.keras.layers.GlobalAveragePooling2D()(efficient_model.output)

    for fc in fclayers:
        x = Dense(fc, activation="relu")(x)
        x = Dropout(0.3)(x)

    output = Dense(classes, activation="softmax")(x)
    model = Model(efficient_model.input, output)
    return model


def create_efficientB0(input_shape, classes=3, fclayers=[2048, 1024], trainable=False, init='imagenet'):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    # Remove fully connected layer and replace
    efficient_model = EfficientNetB0(include_top=False, weights=init, input_shape=input_shape, classes=classes)
    for layer in efficient_model.layers:
        layer.trainable = trainable

    x = tf.keras.layers.GlobalAveragePooling2D()(efficient_model.output)

    for fc in fclayers:
        x = Dense(fc, activation="relu")(x)
        x = Dropout(0.3)(x)

    output = Dense(classes, activation="softmax")(x)
    model = Model(efficient_model.input, output)
    return model



def create_mobileV2(input_shape, classes=3, fclayers=[2048, 1024], trainable=False, init='imagenet'):
    """
    Architecture and adaptation of the VGG16 for our project
    """
    # Remove fully connected layer and replace
    mobile_model = MobileNetV2(include_top=False, weights=init, input_shape=input_shape, classes=classes)
    for layer in mobile_model.layers:
        layer.trainable = trainable

    x = tf.keras.layers.GlobalAveragePooling2D()(mobile_model.output)

    for fc in fclayers:
        x = Dense(fc, activation="relu")(x)
        x = Dropout(0.3)(x)

    output = Dense(classes, activation="softmax")(x)
    model = Model(mobile_model.input, output)
    return model



def compiling(model):
    initial_learning_rate = 0.01
    first_decay_steps = 500
    lr_decayed_fn = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate, first_decay_steps
    )
    optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def loading_checkpoint(model, modelname, warmstart):
    if warmstart != str(0):
        model.load_weights("./logs/" + modelname + "/cpft-" + warmstart + ".ckpt")


def create_model(name, input_shape, classes, fclayers, trainable, init):
    """Create a Keras Sequential models for Computer Vision.

        Args:
            name: name of the model: the alternatives are VGG16, XceptionNet, EfficientNetB0, EfficientNetB3 and MobileV2Net
            input_shape: shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)`
                (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 input channels,
                and width and height should be no smaller than 32.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            fclayers: list of dense layers with the number of neurons
                    to add on top of the pretrained model. e.g. [2048, 1024]
                    for 2 dense layers, 1st with 2048 neurons and the 2nd
                    with 1024 neurons.
            trainable: boolean for finetuning (True) or not (False) the pretrained
                        model
            init: str for initialising the weights. 'imagenet' for using pretrained
                weights on the imagenet dataset. None for no initialisation.
        Returns:
          A `keras.Model` instance.
        """
    if 'vgg' in name.lower():
        return create_vgg16(input_shape, classes, fclayers, trainable, init)
    if 'xception' in name.lower():
        return create_xception(input_shape, classes, fclayers, trainable, init)
    if 'b3' in name.lower():
        return create_efficientB3(input_shape, classes, fclayers, trainable, init)
    if 'b0' in name.lower():
        return create_efficientB0(input_shape, classes, fclayers, trainable, init)
    if 'mobile' in name.lower():
        return create_mobileV2(input_shape, classes, fclayers, trainable, init)
    raise AttributeError(
        'The model' + str(name) + ' is not available. Choice: VGG16, XceptionNet, EfficientNet (B0 or B3) and MobileV2Net')

