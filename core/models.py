from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .batch_norm import BatchNormalization
from .utils import broadcast_iou

flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.45, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


    
def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    """
    Build the entire Yolo v3 Neural Network.
    
    :param size: size of the input image
    :param channels: number of colour channels of the input image
    :param anchors: list of prior boxes (9, 3 boxes for 3 different scales)
    :param masks: indices for the prior boxes
    :param classes: number of avaiable classes
    :param training: whether training of inference
    
    :return: tf.keras.Model instance that represents the entire Yolo v3 network.
    """
    
    x = inputs = Input([size, size, channels], name='input')
    
    # Darknet Network for feature extraction at three different scales
    x_36, x_61, x = DarkNet(name='yolo_darknet')(x)
        
    # Perform conv. operation on 13x13 feature maps.
    x = DetectionNetwork(512, name='yolo_conv_0')(x) 
    # Predict the final output at 13x13 scale
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)
    
    # Upscale 13x13 feature map (x) by two, concatenate with 26x26 feature map (x_61) and 
    # perforn conv.operations
    x = DetectionNetwork(256, name='yolo_conv_1')((x, x_61)) 
    # Predict the final output at 26x26 scale
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)
   
    # Upscale 26x26 feature map (x) by two, concatenate with 52x52 feature map (x_36) and 
    # perforn conv.operations
    x = DetectionNetwork(128, name='yolo_conv_2')((x, x_36)) 
    # Predict the final output at 52x52 scale
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)
    
    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')
    
    # Calculate the bounding boxes and extract the predictions at scale 13x13
    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
             name='yolo_boxes_0')(output_0)
    # Calculate the bounding boxes and extract the predictions at scale 26x26
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    # Calculate the bounding boxes and extract the predictions at scale 52x52
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    # Compute non_max_suppression
    outputs = Lambda(lambda x: non_max_suppression(x, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')

def DarkNet(name='yolo_darknet'):
    """
    53-layer darknet neural network for feature extraction at three different resolutions (13, 26, 52)
    
    :param x: input data
    :param filters: number of filters for the conv. layer
    :param blocks: numer of residual blocks that should be build
    :param strides: size of the stride
    
    :return: Final output of the Yolo network at three different scales
    """
    
    x = inputs=Input([None, None, 3])
    x = ConvLayer(x, 32, 3)
    x = ConvLayer(x, 64, 3, strides=2)
    x = ResidualBlock(x, 32, blocks=1)
    x = ConvLayer(x, 128, 3, strides=2)
    x = ResidualBlock(x, 64, blocks=2)
    x = ConvLayer(x, 256, 3, strides=2)
    x = output_1 = ResidualBlock(x, 128, blocks=8)
    x = ConvLayer(x, 512, 3, strides=2)
    x = output_2= ResidualBlock(x, 256, blocks=8)
    x = ConvLayer(x, 1024, 3, strides=2)
    x = ResidualBlock(x, 512, blocks=4)
    
    
    return tf.keras.Model(inputs, (output_1, output_2, x), name=name)

def DetectionNetwork(filters, name=None):
    """
    A detection network that serves as a intermediate network between darknet and the output network.
    Detection layer applies several convolution operation on the input data. The number of conv. operations
    depends on whether the input consists of conv. images at two different scales. 
    The output of the detection layer is used later for computing the final output of the Yolo network.
    
    :param filters: number of filters for the conv. layers

    """
    def detect(x_in):
        
        if isinstance(x_in, tuple):
            
             inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
             x, x_skip = inputs 
             x = ConvLayer(x, filters, 1)
             x = UpSampling2D(2)(x)
             x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
        
        
        x = ConvLayer(x, filters, 1)
        x = ConvLayer(x, filters * 2, 3)
        x = ConvLayer(x, filters, 1)
        x = ConvLayer(x, filters * 2, 3)
        x = ConvLayer(x, filters, 1)
        
        return Model(inputs, x, name=name)(x_in)

    return detect


    
def YoloOutput(filters, anchors, classes, name=None):
    """
    Computes one regular convolution and one stride-1 convolution for each grid cell.
    The output is the final output of the Yolo network, computed for three different scales (13, 26, 52).
    The output has the shape [batch_size, grid, grid, anchors*(classes+(x, y, w, h) + obj)]
    
    :param filters: number of filters for the conv layer
    :param anchors: number of prior boxes
    :param classes: number of classes 
    
    :return: Final output of the Yolo network at three different scales
    """
    
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = ConvLayer(x, filters * 2, 3)
        
        # Compute for each grid cell a prediction vector of size (anchors* (classes+5))
        x = ConvLayer(x, anchors * (classes + 5), 1, batch_norm=False)
        
        # Reshape the predictions to the shape [batch_size, grid, grid, anchors*(classes+(x, y, w, h) + obj)]
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output   


def yolo_boxes(pred, anchors, classes):
    """
    Calculate the coordinates of the predicted bounding boxes according to predicted values
    tx, ty, tw, th.
    
    :param pred: Prediction of Yolo at three different scales (grid_size). 
    The shape is: [batch_size, grid_size, grid_size, anchors, (x, y, w, h, obj, num_classes)]
    
    :param anchors: number of prior boxes
    :param classes: number of classes 
    
    :return: bbox, coordinates of the predicted bbox [batch, grid, grid, anchors (x1, y1, x2, y2)]
    :return: objectness [batch, grid, grid, anchors 1]
    :return: class_probs [batch, grid, grid, anchors, num_classes]
    :return: pred_box [batch, grid, grid, anchors, (σ(tx), σ(ty), tw, th)]
    """
    
    # Get the size of the feature map (13, 26 or 52)
    grid_size = tf.shape(pred)[1] 
    
    # Split the prediction vector at the last dim. to the the unique predictions
    # box_xy: unscaled (x, y) centroids  [batch_size, grid_size, grid_size, anchors, (tx, ty)]
    # box_wh: unscaled width and hight of the box box_wh: [batch_size, grid_size, grid_size, anchors, (tw, th)]
    # objectness: Is there an object in the grid cell? [batch_size, grid_size, grid_size, anchors, 1]
    # class_probs: Probability values that a grid cell holds a particular object: [batch_size, grid_size, grid_size, anchors, classes]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)
    
    # box_xy: (σ(tx), σ(ty)) cordinates of the bbox relative to the grid cell
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    # Save ((σ(tx), σ(ty))) and tw, th predictions for later
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  
        
    # Basically: bx = σ(tx) + cx, by = σ(ty) + cy, coordinates relative to the entire image
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32) #box_xy: [batch_size, grid_size, grid_size, anchors, 2]
        
    # Compute the actual width and hight 
    box_wh = tf.exp(box_wh) * anchors

    # Compute (x1, y1), (x2, y2) coordinates of the bbox relative to the entire image 
    box_x1y1 = box_xy - box_wh / 2 # top left corner
    box_x2y2 = box_xy + box_wh / 2 # bottom right corner
    
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
     
    return bbox, objectness, class_probs, pred_box

 
def non_max_suppression(outputs, masks, classes):
    """
    This operation performs non_max_suppression on the inputs per batch, across all classes. 
    Prunes away boxes that have high intersection-over-union (IOU) overlap with previously selected boxes. 
    Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates 
    of any diagonal pair of box corners. The output of this operation is the final boxes, scores and 
    classes tensor returned after performing non_max_suppression.
    
    :param pred: Prediction of Yolo at three different scales (grid_size). 
    The shape is: [batch_size, grid_size, grid_size, anchors, (x, y, w, h, obj, num_classes)]
    
    :param outputs: (bounding boxes, objectness, class probabilities) at three different scales at the same time
    :param masks: indices for the anchors at three different scales
    
    :return: boxes
    :return: scores
    :return: classes
    :return: valid_detections
    """
    
    # outputs: [batch_size, grid_size, grid_size, 3, 4], [batch_size, grid_size, grid_size, 3, 1], [batch_size, grid_size, grid_size, 3, 80]
 
    # boxes, confidence_score, class-types
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    
    # b : [batch_size, grid_size*grid_size*3, 4]
    # c : [batch_size, grid_size*grid_size*3, 1]
    # t : [batch_size, grid_size*grid_size*3, 80]

    bbox = tf.concat(b, axis=1) # [batch_size, 10647, 4]
    confidence = tf.concat(c, axis=1) # [batch_size, 10647, 1]
    class_probs = tf.concat(t, axis=1) # [batch_size, 10647, 80]
    
    scores = confidence * class_probs
    
    boxes_reshaped=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)) # [batch_size, 10647, 1, 4]
    scores_reshaped= tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])) # [batch_size, 10647, 80]
    
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes_reshaped, 
        scores=scores_reshaped,
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections
    
    
def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        """
       

        :param y_true: labels at three different scales [batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, label)]
        :param y_pred: prediction of Yolo at 3 different scales [batch_size, grid, grid, anchors, 85)]
        
        :return: yolo_loss Sum of the losses for a prediction at a certain scale
        """

        # pred_box: [batch_size, grid_size, grid_size, anchors, (x1, y1, x2, y2)]
        # pred_xywh: [batch_size, grid_size, grid_size, anchors, (x, y, w, h)]
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
          
        pred_xy = pred_xywh[..., 0:2] # [None, grid_size, grid_size, 3, (σ(tx), σ(ty))]
        pred_wh = pred_xywh[..., 2:4] # [None, grid_size, grid_size, 3, (tw,th)]
        

        #Split the tensors along the last axis
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        
        #true_box: [None, grid_size, grid_size, 3, (x1, y1, x2, y2)]
        #true_obj: [None, grid_size, grid_size, 3, 1]
        #true_class_idx: [None, grid_size, grid_size, 3, 1]
        
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2 # true (x,y) center
        true_wh = true_box[..., 2:4] - true_box[..., 0:2] # true w & h

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. Inverting the pred box equations (get centroid of the bbox relative to the grid cell, basically σ(tx), σ(ty))
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2) # [grid_size, grid_size, 1, 2]
        true_xy = true_xy * tf.cast(grid_size, tf.float32) -  tf.cast(grid, tf.float32)  
        
        # Inverting the true width and height, basically getting tw, th    
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1) # [batch_size, grid, grid, anchors]
    
        # ignore false positive when iou is over threshold
        # The `ignore_mask` is used to make sure we only penalize when the current box doesn’t have much overlap with the ground truth box
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
    

def ResidualBlock(x, filters, blocks, strides=1, batch_norm=True):
    """
    A residual block that performs a residual convolution
    
    :param x: input data
    :param filters: number of filters for the conv. layer
    :param blocks: numer of residual blocks that should be build
    :param strides: size of the stride
    
    :return: Final output of the Yolo network at three different scales
    """
    for _ in range(0, blocks):
        prev = x
        x = ConvLayer(x, filters, size=1, strides=1, batch_norm=True)
        x = ConvLayer(x, filters*2, size=3, strides=1, batch_norm=True)
        x = Add()([prev, x])
    
    return x

def ConvLayer(x, filters, size, strides=1, batch_norm=True):
    """
    Regular convolution layer.
    
    :param x: input data
    :param filters: number of filters for the conv. layer
    :param size: size of the convolution matrix
    :param strides: size of the stride
    
    :return: Convoluted image
    """
    
    
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x
    
     

