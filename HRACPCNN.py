from operator import truediv
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Conv1D,Conv2D, Conv3D, Flatten, Dense, Reshape, Lambda
from tensorflow.keras.layers import Dropout, Input,dot,Activation,MaxPool1D,add,BatchNormalization,MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from collections import Counter

# from MDGCN import trainMDGCN
tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)

import time
from plotly.offline import init_notebook_mode
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
time_start=time.time()
## Data
## GLOBAL VARIABLES
# dataset = 'IP'
# test_ratio = 0.9

# dataset = 'KSC'
# test_ratio = 0.94

dataset = 'UP'
test_ratio = 0.7

train_val_ratio = 1
train_ratio = 1-test_ratio
windowSize = 11
if dataset == 'UP':
    componentsNum = 30
elif dataset == 'UH':
    componentsNum = 50 if test_ratio >= 0.99 else 25
elif dataset == 'IP':
    componentsNum = 140
elif dataset == 'KSC':
    componentsNum = 120
else:
    componentsNum = 30
drop = 0.4

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})
## define a series of data progress function
def loadData(name):
    data_path = os.path.join(os.getcwd(),'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'UH':
        data = sio.loadmat(os.path.join(data_path, 'HoustonU.mat'))['houstonU'] # 601*2384*50
        labels = sio.loadmat(os.path.join(data_path, 'HoustonU_gt.mat'))['houstonU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    return data, labels
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,stratify=y)
    return X_train, X_test, y_train, y_test
def applyPCA(X, numComponents=140):
    newX = np.reshape(X, (-1, X.shape[2]))
    print(newX.shape)
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca, pca.explained_variance_ratio_
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype="float16")
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def createPatches(X, y, windowSize=25, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]),dtype="float16")
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]),dtype="float16")
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def SBNL(X,numComponents):
    X_copy1 = np.zeros((X.shape[0],X.shape[1],X.shape[2]))
    half1 = int(numComponents/2)
    for j in range(0,half1-2):
        X_copy1[:,:,2*j] = X[:,:,j]
        X_copy1[:,:,2*j+1] = X[:,:,numComponents-j-2]
    #X_copy1[:,:,198] = X[:,:,99]
    X_copy1[:,:,102] = X[:,:,102]
    #X_copy1[:, :, 174] = X[:, :, 87]
    X = X_copy1
    return X

def non_local_block(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        # phi = Conv2D(channels, (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)#内积函数
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])#compression=1

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    # ip_ = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    # ip__ = Conv2D(channels, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    # y = tf.concat([y,ip_,ip__],3)
    y = _convND(y, rank, channels)#1*1*C操作

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y

def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x1 = Conv2D(channels, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x1)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x

X, y = loadData(dataset)
#X = SBNL(X,200)
X = SBNL(X,103)
#X = SBNL(X,176)
X,pca,ratio = applyPCA(X,numComponents=componentsNum)
# X = infoChange(X,componentsNum) # channel-wise shift
# X = SBNL(X,componentsNum)
X, y = createPatches(X, y, windowSize=windowSize)
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

## Train
Xtrain = Xtrain.reshape(-1, windowSize, windowSize, componentsNum, 1)
ytrain = to_categorical(ytrain)
for col in range(ytrain.shape[1]):
    b = Counter(ytrain[:,col])
    print(b)
Xvalid, Xtest, yvalid, ytest = splitTrainTestSet(Xtest, ytest, (test_ratio-train_ratio/train_val_ratio)/test_ratio)
Xvalid = Xvalid.reshape(-1, windowSize, windowSize, componentsNum, 1)
yvalid = to_categorical(yvalid)
if dataset == 'UP':
    output_units = 9
elif dataset == 'UH':
    output_units = 20
elif dataset == 'KSC':
    output_units = 13
else:
    output_units = 16
## implementation of covariance pooling layers
def cov_pooling(features):
    shape_f = features.shape.as_list()
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]),2)
    centers_batch = tf.reshape(centers_batch, [-1, 1, shape_f[2]])
    centers_batch = tf.tile(centers_batch, [1, shape_f[1], 1])
    tmp = tf.subtract(features, centers_batch)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    features_t = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    trace_t = tf.compat.v1.trace(features_t)
    trace_t = tf.reshape(trace_t, [-1, 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    trace_t = 0.0001*tf.compat.v1.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)
def feature_vector(features):
    shape_f = features.shape.as_list()
    feature_upper = tf.linalg.band_part(features,0,shape_f[2])
    return feature_upper
def bn_prelu(X):
    X = BatchNormalization(epsilon=1e-5)(X)
    X = Activation('Mish')(X)
    return X

## input layer
input_layer = Input((windowSize, windowSize, componentsNum, 1))
## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu',padding='same')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu',padding='same')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu',padding='same')(conv_layer2)
# print(conv_layer3._keras_shape)
conv3d_shape = conv_layer3.shape
conv_layer3_ = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)

conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(conv_layer3_)
conv_layer4 = bn_prelu(conv_layer4)

conv_layer5 = Conv2D(filters=64, kernel_size=(1,1))(conv_layer4)
conv_layer5_ = tf.concat([conv_layer5,conv_layer4,conv_layer5],0)
conv_layer5_ = bn_prelu(conv_layer5_)
conv_layer6 = Conv2D(filters=192, kernel_size=(1,3), padding='same')(conv_layer5_)
conv_layer6 = bn_prelu(conv_layer6)
conv_layer7 = Conv2D(filters=192, kernel_size=(3,1), padding='same')(conv_layer6)
conv_layer7 = bn_prelu(conv_layer7)
conv_layer8 = Conv2D(filters=192, kernel_size=(3,3), padding='same')(conv_layer7)
conv_layer8 = bn_prelu(conv_layer8)
conv_layer9 = Conv2D(filters=192, kernel_size=(5,5), padding='same')(conv_layer8)
conv_layer9 = bn_prelu(conv_layer9)
conv_layer9_ = Conv2D(filters=192, kernel_size=(7,7), padding='same')(conv_layer8)
conv_layer9_ = bn_prelu(conv_layer9_)
conv_layer10 = Conv2D(filters=64, kernel_size=(1,1))(conv_layer9_)
conv_layer11 = conv_layer10 + conv_layer4
#
conv_layer11 = non_local_block(conv_layer11,compression=1)
conv2d_shape = conv_layer11.shape

conv_layer4 = Reshape((conv2d_shape[1] * conv2d_shape[2], conv2d_shape[3]))(conv_layer4)
conv2d_shape = conv_layer4.shape
cov_pooling_layer1 = Lambda(cov_pooling,output_shape=(conv2d_shape[2],conv2d_shape[2]),mask=None,arguments=None)(conv_layer4)
cov_pooling_layer2 = Lambda(feature_vector,output_shape=(conv2d_shape[2],conv2d_shape[2]),mask=None,arguments=None)(cov_pooling_layer1)
flatten_layer = Flatten()(cov_pooling_layer2)

## fully connected layers
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
# compiling the model
adam = Adam(lr=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# checkpoint
filepath = "best-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
start = time.time()
history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=50, validation_data=(Xvalid,yvalid), callbacks=callbacks_list)  #,validation_split=(1/3)
end = time.time()
train_time = end - start
print(train_time)
plt.figure(figsize=(7,7))
plt.grid()
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'], loc='upper right')
plt.savefig("loss_curve.pdf")
plt.show()
plt.figure(figsize=(5,5))
plt.ylim(0,1.1)
plt.grid()
plt.plot(history.history['accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'])
plt.savefig("acc_curve.pdf")
plt.show()

## Test
# load best weights
model.load_weights("best-model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
Xtest = Xtest.reshape(-1, windowSize, windowSize, componentsNum, 1)
# Xtest.shape
ytest = to_categorical(ytest)
# ytest.shape
Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)
classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def reports (X_test,y_test,name):
    start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    end = time.time()
    print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                        'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'UP':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    elif name == 'UH':
        target_names = ['Healthy grass','Stressed grass','Artificial turf','Evergreen trees', 'Deciduous trees','Bare earth','Water',
                        'Residential buildings','Non-residential buildings','Roads','Sidewalks','Crosswalks','Major thoroughfares','Highways',
                       'Railways','Paved parking lots','Unpaved parking lots','Cars','Trains','Stadium seats']

    elif name == 'KSC':
        target_names = ['Srub','Willow swamp','CP hammock','Slash pine','Oak/Broadleaf','Hardwood',
                        'Swamp','Graminoid','Spartina marsh','Cattail marsh','Salt marsh','Mud flats',
                        'Water']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100
classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset)
classification = str(classification)
confusion1 = str(confusion)
file_name = "classification_report.txt"
with open(file_name, 'w') as x_file:
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion1))