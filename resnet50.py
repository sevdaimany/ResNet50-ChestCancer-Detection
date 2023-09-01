from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, Add, Dense, Activation
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import get_file


def identity_block(X, filters):
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    # first component
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    # second component
    X = Conv2D(filters=F2, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer=glorot_uniform(seed=0))(X) 
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    # third component
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1,1), padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    
    return X


def convolutional_block(X, filters, s):
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    # first component
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    # second component
    X = Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding="same", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    # third component
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding="valid", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def ResNet50(input_shape=(224, 224, 3),
            weights=None,
            classes=1000,
            include_top=True):
    
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    
    # stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2,2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)
    
    # stage 2
    X = convolutional_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])
    
    # stage 3
    X = convolutional_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    
    
    # stage 4
    X = convolutional_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    
    # stage 5
    X = convolutional_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    
    
    if include_top:
        X = AveragePooling2D(pool_size=(2,2), padding='same')(X)
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input, outputs=X)
    
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
            
        
        model.load_weights(weights_path)
    
    return model