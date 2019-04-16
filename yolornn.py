from keras.models import Sequential, Model
from keras.layers import Average, Concatenate, UpSampling2D, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import pickle, numpy as np
import os, cv2
from keras.engine.topology import Layer
from net.netparams import YoloParams
from net.utils import draw_boxes
from crfrnn_layer import CrfRnnLayer

def process_outs(b, s, c):
    
    b_p = b
    # Expand dims of scores and classes so we can concat them 
    # with the boxes and have the output of NMS as an added layer of YOLO.
    # Have to do another expand_dims this time on the first dim of the result
    # since NMS doesn't know about BATCH_SIZE (operates on 2D, see 
    # https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression) 
    # but keras needs this dimension in the output.
    s_p = K.expand_dims(s, axis=-1)
    c_p = K.expand_dims(c, axis=-1)
    
    output_stack = K.concatenate([b_p, s_p, c_p], axis=1)

    # if output.size == 0:
    #         return [np.array([])]*4

    #     boxes = output[:,:4]
    #     scores = output[:,4]
    #     label_idxs = output[:,5].astype(int)

    #     labels = [YoloParams.CLASS_LABELS[l] for l in label_idxs]

    return K.expand_dims(output_stack, axis=0)

class YOLODetectionLayer(Layer):
    def __init__(self, detection_threshold, num_classes, max_boxes, nms_threshold, **kwargs):
        self.detection_threshold = detection_threshold
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.nms_threshold = nms_threshold
        super(YOLODetectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YOLODetectionLayer, self).build(input_shape)

    def call(self, inputs): #input is 13x13x50x(4+1+nb_class)
        # decode boxes
        # need to convert b's from GRID_SIZE units into IMG coords. Divide by grid here. 
        b_xy = (K.sigmoid(inputs[..., 0:2]) + YoloParams.c_grid[0]) / YoloParams.GRID_SIZE
        b_wh = (K.exp(inputs[..., 2:4])*YoloParams.anchors[0]) / YoloParams.GRID_SIZE
        b_xy1 = b_xy - b_wh / 2.
        b_xy2 = b_xy + b_wh / 2.
        boxes = K.concatenate([b_xy1, b_xy2], axis=-1)
        
        # filter out scores below detection threshold
        scores_all = K.sigmoid(inputs[..., 4:5]) * K.softmax(inputs[...,5:])
        indicator_detection = scores_all > self.detection_threshold
        scores_all = scores_all * K.cast(indicator_detection, np.float32)

        # compute detected classes and scores
        classes = K.argmax(scores_all, axis=-1)
        scores = K.max(scores_all, axis=-1)

        # flattened tensor length
        S2B = YoloParams.GRID_SIZE*YoloParams.GRID_SIZE*YoloParams.NUM_BOUNDING_BOXES

        # flatten boxes, scores for NMS
        flatten_boxes = K.reshape(boxes, shape=(S2B, 4))
        flatten_scores = K.reshape(scores, shape=(S2B, ))
        flatten_classes = K.reshape(classes, shape=(S2B, ))

        inds = []

        # apply multiclass NMS 
        for c in range(self.num_classes):

            # only include boxes of the current class, with > 0 confidence
            class_mask = K.cast(K.equal(flatten_classes, c), np.float32)
            score_mask = K.cast(flatten_scores > 0, np.float32) 
            mask = class_mask * score_mask
            
            # compute class NMS
            nms_inds = tf.image.non_max_suppression(
                    flatten_boxes, 
                    flatten_scores*mask, 
                    max_output_size=self.max_boxes, 
                    iou_threshold=self.nms_threshold
                )
            
            inds.append(nms_inds)

        # combine winning box indices of all classes 
        selected_indices = K.concatenate(inds, axis=-1)
        
        # gather corresponding boxes, scores, class indices
        selected_boxes = K.gather(flatten_boxes, selected_indices)
        selected_scores = K.gather(flatten_scores, selected_indices)
        selected_classes = K.gather(flatten_classes, selected_indices)

        # mask = np.zeros((YoloParams.INPUT_SIZE, YoloParams.INPUT_SIZE, 1))
        # draw_boxes(mask, [selected_boxes, selected_scores, selected_classes])
        # return np.expand_dims(mask,axis=0)

        mask = tf.zeros((1, YoloParams.INPUT_SIZE, YoloParams.INPUT_SIZE, 1))
        mask = tf.image.draw_bounding_boxes(mask, selected_boxes)

        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], YoloParams.INPUT_SIZE, YoloParams.INPUT_SIZE, 1)

input_image = Input(shape=(416, 416, 3))

# Layer 1
x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2 - 5
for i in range(0,4):
    x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(i+2))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

for i in range(0,2):
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(i+7))(x)
    x = LeakyReLU(alpha=0.1)(x)

grid_h, grid_w = YoloParams.GRID_SIZE,YoloParams.GRID_SIZE
nb_class = 6
nb_box = YoloParams.NUM_BOUNDING_BOXES
true_boxes = Input(shape=(1, 1, 1, nb_box , 4))
output = Conv2D(nb_box * (4 + 1 + nb_class), 
                (1,1), strides=(1,1), 
                padding='same', 
                name='DetectionLayer', 
                kernel_initializer='lecun_normal')(x)
output = Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))(output)
output = Lambda(lambda args: args[0])([output, true_boxes])

# yolo mask
mask = YOLODetectionLayer(nms_threshold = YoloParams.NMS_THRESHOLD, max_boxes = 50, num_classes = YoloParams.NUM_CLASSES, detection_threshold=YoloParams.DETECTION_THRESHOLD)(output)

model = Model([input_image, true_boxes], mask)
model.summary()

image = cv2.imread('../yolov2/data/fibroblast_nucleopore_class/2018/JPEGImages/010037.jpg')
img = cv2.resize(image, (416, 416))
img = img / 255.
img = img[:,:,::-1]
img = np.expand_dims(img, 0)
dummy = np.zeros(shape=(1, 1, 1, 1, nb_box , 4))
out = model.predict([img, dummy])
print(out.shape)

# # crf layer
# # get feature maps
# pool1 = model.get_layer('max_pooling2d_1')
# up1 = UpSampling2D()(pool1.output)
# pool2 = model.get_layer('max_pooling2d_2')
# up2 = UpSampling2D()(pool2.output)
# up2 = UpSampling2D()(up2)
# pool3 = model.get_layer('max_pooling2d_3')
# up3 = UpSampling2D()(pool3.output)
# up3 = UpSampling2D()(up3)
# up3 = UpSampling2D()(up3)
# pool4 = model.get_layer('max_pooling2d_4')
# up4 = UpSampling2D()(pool4.output)
# up4 = UpSampling2D()(up4)
# up4 = UpSampling2D()(up4)
# up4 = UpSampling2D()(up4)

# concat = Concatenate()([up1, up2, up3, up4])
# U = Input(shape=(416,416, 1))
# crf = CrfRnnLayer(image_dims=(416, 416),
#                          num_classes=nb_class,
#                          theta_alpha=160.,
#                          theta_beta=3.,
#                          theta_gamma=3.,
#                          num_iterations=10,
#                          name='crfrnn')([U, concat])

# yolornnmodel = Model([U, input_image], crf)
# yolornnmodel.summary()