import os
import sys
from pathlib import Path
import numpy as np

# Utils
filepath = Path(__file__).resolve().parent
# sys.path.append( os.path.abspath(filepath/'../ml') )
from ml.keras_utils import r2_krs
from ml.data import extract_subset_fea


try:
    import tensorflow as tf
    if int(tf.__version__.split('.')[0]) < 2:
        import keras
        from keras.models import load_model
        from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
        from keras.utils import plot_model
    else:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
        from tensorflow.keras.utils import plot_model

        from tensorflow import keras
        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
        from tensorflow.keras import layers
        from tensorflow.keras import optimizers
        from tensorflow.keras.optimizers import SGD, Adam
        from tensorflow.keras.models import Sequential, Model
except:
    print('Could not import tensorflow.')


def clr_keras_callback(mode=None, base_lr=1e-4, max_lr=1e-3, gamma=0.999994):
    """ Creates keras callback for cyclical learning rate. """
    from . cyclical_learning_rate import CyclicLR
    if mode == 'trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif mode == 'trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif mode == 'exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
    return clr


def model_callback_def(outdir, ref_metric='val_loss', **clr_kwargs):
    """ Required for lrn_crv.py """
    checkpointer = ModelCheckpoint(str(outdir/'model_best.h5'), monitor='val_loss',
                                   verbose=0, save_weights_only=False,
                                   save_best_only=True)
    csv_logger = CSVLogger(outdir/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor=ref_metric, factor=0.75, patience=15,
                                  verbose=1, mode='auto', min_delta=0.0001,
                                  cooldown=3, min_lr=0.000000001)
    early_stop = EarlyStopping(monitor=ref_metric, patience=20, verbose=1,
                               mode='auto')

    if bool(clr_kwargs):
        clr = clr_keras_callback(**clr_kwargs)
        return [checkpointer, csv_logger, early_stop, reduce_lr, clr]

    return [checkpointer, csv_logger, early_stop, reduce_lr]


def nn_reg0_model_def(input_dim: int,
                      batchnorm: bool=False, dr_rate: float=0.2,
                      learning_rate: float=0.001, opt_name: str='adam',
                      **kwargs):
    """
    Create the Keras model. This is func is required in lrn_crv.py.
    **kwargs is used to ignore irrelevant arguments passed hps_set
    of keras-tuner.
    """
    initializer = 'he_uniform'

    units = [1000, 1000, 500, 250, 125] # fair
    inputs = keras.layers.Input(shape=(input_dim,), name='inputs')

    x = layers.Dense(units[0], kernel_initializer=initializer)(inputs)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units[1], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units[2], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units[3], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units[4], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    outputs = layers.Dense(1, activation='relu', name='outputs')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    if opt_name.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    return model


# ------------------------------------------------------------
def data_prep_nn0_def(xdata):
    """
    This func prepares the dataset for keras model.
    """
    xdata = np.asarray(xdata)
    x_dct = {'inputs': xdata}
    return x_dct


# ----------------------------------------------------------------
def nn_reg1_model_def(in_dim_ge: int, in_dim_dd: int,
                      batchnorm: bool=False, dr_rate: float=0.2,
                      learning_rate: float=0.001, opt_name: str='adam',
                      **kwargs):
    """
    Create the Keras model. This is func is required in lrn_crv.py.
    **kwargs is used to ignore irrelevant arguments passed hps_set
    of keras-tuner.
    """
    initializer = 'he_uniform'

    # ---------------------
    # GE
    in_ge = Input(shape=(in_dim_ge,), name='in_ge')
    units_ge = [800, 500]  # fair (July2020)

    x = layers.Dense(units_ge[0], kernel_initializer=initializer, name='g_dense_0')(in_ge)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units_ge[1], kernel_initializer=initializer, name='g_dense_1')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    out_ge = layers.Dropout(dr_rate)(x)

    ge = Model(inputs=in_ge, outputs=out_ge, name='out_ge')

    # ---------------------
    # DD
    in_dd = Input(shape=(in_dim_dd,), name='in_dd')
    units_dd = [995, 700]  # fair (July2020)

    x = layers.Dense(units_dd[0], kernel_initializer=initializer, name='d_dense_0')(in_dd)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units_dd[1], kernel_initializer=initializer, name='d_dense_1')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    out_dd = layers.Dropout(dr_rate)(x)

    dd = Model(inputs=in_dd, outputs=out_dd, name='out_dd')

    # ---------------------
    # Merge towers
    mrg = layers.concatenate([ge.output, dd.output], axis=1)
    units_mrg = [500, 250, 125]  # fair

    x = layers.Dense(units_mrg[0], kernel_initializer=initializer)(mrg)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units_mrg[1], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    x = layers.Dense(units_mrg[2], kernel_initializer=initializer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dr_rate)(x)

    # ---------------------
    # Output
    outputs = layers.Dense(1, activation='relu', name='outputs')(x)

    # ---------------------
    # Input --> Output
    model = Model(inputs=[in_ge, in_dd], outputs=[outputs])

    if opt_name.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    return model


# ------------------------------------------------------------
def data_prep_nn1_def(xdata):
    """
    This func prepares the dataset for keras model.
    """
    x_ge = extract_subset_fea(xdata, fea_list=['ge'], fea_sep='_')
    x_dd = extract_subset_fea(xdata, fea_list=['dd'], fea_sep='_')
    x_ge = np.asarray(x_ge)
    x_dd = np.asarray(x_dd)
    x_dct = {'in_ge': x_ge, 'in_dd': x_dd}
    return x_dct
