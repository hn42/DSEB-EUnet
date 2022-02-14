
backend = None
layers = None
models = None
keras_utils = None
from keras_applications import get_submodules_from_kwargs



# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(x):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(x)


    return wrapper


def DecoderTransposeX2Block_my3(filters, stage, use_batchnorm=False,input_tensor=None,skip=None,type='bos'):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3
    bn_axis=3
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        name=transp_name,
        use_bias=not use_batchnorm,
    )(input_tensor)

    if use_batchnorm:
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

    x = layers.Activation('relu', name=relu_name)(x)
    if skip is not None and 'nor' in type:
        x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
    elif  skip is not None and 'DSEB' in type:
        shape_skip = backend.int_shape(skip)
        # skip__ = layers.Conv2D(shape_skip[3]//4, (3, 3), padding='same')(skip)
        if '3x3' in type:
            skip__ = layers.DepthwiseConv2D(kernel_size=3,
                                       strides=1,
                                       activation=None,
                                       use_bias=False,
                                       padding='same',)(skip)
        if '5x5' in type:
            skip__ = layers.DepthwiseConv2D(kernel_size=5,
                                       strides=1,
                                       activation=None,
                                       use_bias=False,
                                       padding='same',)(skip)
        if '7x7' in type:
            skip__ = layers.DepthwiseConv2D(kernel_size=7,
                                       strides=1,
                                       activation=None,
                                       use_bias=False,
                                       padding='same',)(skip)

        skip__ = layers.Conv2D(1, (1, 1), padding='same')(skip__)
        # concat = layers.Concatenate(axis=concat_axis, name=concat_name)([x__, skip__])
        skip__ = layers.Activation('sigmoid',name='sig_'+str(stage))(skip__)
        my_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=3),
                                  arguments={'repnum': shape_skip[3]})(
            skip__)
        y = layers.multiply([my_repeat, skip])
        x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, y])
    elif  skip is not None and 'se' in type:
        shape_skip = backend.int_shape(skip)
        if '3x3' in type:
            skip__ = layers.Conv2D(1, (3, 3), padding='same')(skip)
        elif '5x5' in type:
            skip__ = layers.Conv2D(1, (5, 5), padding='same')(skip)
        elif '7x7' in type:
            skip__ = layers.Conv2D(1, (7, 7), padding='same')(skip)
        else:
            skip__ = layers.Conv2D(1, (1, 1), padding='same')(skip)
        # concat = layers.Concatenate(axis=concat_axis, name=concat_name)([x__, skip__])
        skip__ = layers.Activation('sigmoid',name='sig_'+str(stage))(skip__)
        my_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=3),
                                  arguments={'repnum': shape_skip[3]})(
            skip__)
        y = layers.multiply([my_repeat, skip])
        x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, y])
    elif skip is not None and 'wb' in type:
        skip__ = None
        shape_skip = backend.int_shape(skip)
        skip_p1 = layers.Conv2D(1, (1, 1), padding='same')(skip)
        skip_p2 = layers.Conv2D(shape_skip[-1], (1, 1), padding='same')(skip)
        skip_p2 = layers.Conv2D(1, (3, 3), padding='same')(skip_p2)
        skip_p3 = layers.Conv2D(shape_skip[-1], (1, 1), padding='same')(skip)
        skip_p3 = layers.Conv2D(1, (5, 5), padding='same')(skip_p3)
        skip__ = layers.add([skip_p1, skip_p2,skip_p3])
        # concat = layers.Concatenate(axis=concat_axis, name=concat_name)([x__, skip__])
        skip__ = layers.Activation('sigmoid',name='sig_'+str(stage))(skip__)
        my_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=3),
                                  arguments={'repnum': shape_skip[3]})(
            skip__)
        y = layers.multiply([my_repeat, skip])
        x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, y])

    x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)
    return x

def build_dseb_eunet(
        backbone,
        skip_connection_layers,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
type='bos'
):

    decoder_use_batchnorm = True,
    input_ = backbone.input
    x = backbone.output
    decoder_filters = [256, 128, 64, 32, 16]
    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    input_boyut=backend.int_shape(input_)[1]
    output=[]
    for i in range(5):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = DecoderTransposeX2Block_my3(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm,input_tensor=x, skip=skip,type=type)

        out=x
        if 'mout' in type:
            boyut = backend.int_shape(out)[1]
            # if input_boyut
            kernal=input_boyut//boyut
            if kernal>1:
                out = layers.UpSampling2D((kernal, kernal), interpolation='nearest')(out)
        # model head (define number of output classes)
            output.append(out)

    if 'mout' in type:

        out = layers.Concatenate(axis=3)(output)
        out = Conv3x3BnReLU(16, use_batchnorm)(out)
        out = layers.Conv2D(
            filters=classes,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name='final_conv',
        )(out)
        x = layers.Activation(activation, name=activation)(out)


    else:
        out = layers.Conv2D(
            filters=classes,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name='final_conv',
        )(out)
        out = layers.Activation(activation, name=activation)(out)
        x=out
    # create keras model instance


    model = models.Model(input_, [x])

    return model
# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------
import tensorflow as tf
def DSEB_EUNET_MODEL(
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        type='bos',
        **kwargs
):
    _KERAS_BACKEND = tf.keras.backend
    _KERAS_LAYERS = tf.keras.layers
    _KERAS_MODELS = tf.keras.models
    _KERAS_UTILS = tf.keras.utils

    kwargs['backend'] = _KERAS_BACKEND
    kwargs['layers'] = _KERAS_LAYERS
    kwargs['models'] = _KERAS_MODELS
    kwargs['utils'] = _KERAS_UTILS

    backbone_name='efficientnetb0'
    encoder_weights = 'imagenet',
    decoder_filters = (256, 128, 64, 32, 16),

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    import efficientnet.model as eff
    backbone = eff.EfficientNetB0(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False,
        **kwargs,
    )
    efficientnetb0_skip_layer= ['block6a_expand_activation', 'block4a_expand_activation',
                       'block3a_expand_activation', 'block2a_expand_activation']
    model = build_dseb_eunet(
        backbone=backbone,
        skip_connection_layers=efficientnetb0_skip_layer,
        classes=classes,
        activation=activation,
        type=type
    )
    return model


