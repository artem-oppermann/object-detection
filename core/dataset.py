import tensorflow as tf
from absl.flags import FLAGS

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    '''
    Outputs a label tensor y_true_out for three different scales. Only for one anchor (according to best_anchor_idx) the values are not zero.
    
    @param y_true: [batch_size, num_objects_per_image, (x1, y1, x2, y2, label, best_anchor_idx)]
    @param grid_size: size of the feature maps
    @param anchor_idxs: indices of the prior box sizes for three different scales ([6, 7, 8], [3, 4, 5], [0, 1, 2])
    
    return: tf.data.Dataset instance
    '''
    N = tf.shape(y_true)[0] # How many batches

    # zero-tensor to be filled later: [batch_size, grid_size, grid_size, num_anchors, [x, y, w, h, obj, class]]
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    
    # Iterate over batches
    for i in tf.range(N):
        
        # Iterate over objects in each image
        for j in tf.range(tf.shape(y_true)[1]):
   
            if tf.equal(y_true[i][j][2], 0): # x2 = 0? --> skip
                continue
            
            # Does index of best anchor match an index in the anchor mask
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32)) 

            # Fille the zero-tensor with values (bboxes, objectness, labels) according the the matching anchor
            if tf.reduce_any(anchor_eq): # "logical or" across elements
                box = y_true[i][j][0:4] # take (x1, y1, x2, y2)
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2 # compute center of the box

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32) # Index of "True" in anchor_eq
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)
                    
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]) 
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]) # [x1, y1, x2, y2, 1, label]
                idx += 1


    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    '''
    
    @param y_train: [batch_size, num_objects_per_image, (x1, y1, x2, y2, label)]
    @param anchors: list of scales of the 9 piror bboxes at three different scales
    @param anchor_masks: indices of the prior box sizes for three different scales
    @param size: size of the image that goes into the network
    
    return: tf.data.Dataset instance
    '''
    #y_train: [batch_size,  Number of objects in image, (x1, y1, x2, y2, label)]
    y_outs = []
    grid_size = size // 32 # 13 (min. grid size)

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    # Calculate the area for all 9 anchors
    anchor_area = anchors[..., 0] * anchors[..., 1]
    
    # Calculate with and hight of the target bboxes
    box_wh = y_train[..., 2:4] - y_train[..., 0:2] # [batch_size, num_objects_per_image, (w,h)]
    box_wh_expand_dims=tf.expand_dims(box_wh, -2) # [batch_size, num_objects_per_image, 1, (w,h)]
    
    # Expand the third dimension by the number of anchors
    box_wh = tf.tile(box_wh_expand_dims,
                     (1, 1, tf.shape(anchors)[0], 1)) # [batch_size, num_objects_per_image, 9 , 2]
    
    # Compute IoU for each Anchor (and scale ) with the groud truth bb
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection) # [batch_size, num_objects_per_image, 9]
    # Select the id of the anchor with the best IoU with the target bbox
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32) # [batch_size, num_objects_per_image, index of anchor with highest IoU]
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)  # [batch_size, num_objects_per_image, index of anchor with highest IoU ,1]
    
    y_train = tf.concat([y_train, anchor_idx], axis=-1) # [batch_size, num_objects_per_image, (x1, y1, x2, y2, label, anchor_idx)]
    
    # Iterate over all scales (13x13, 26x26, 52x52)
    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}




def load_tfrecord_dataset(file_name, class_file, size=416):
    '''
    Load the tfrecord files.
    
    @param file_pattern: name of the tfrecord file
    @param class_file: file that contains the names of the classes
    @param size: size of the input image for the network
    
    return: tf.data.Dataset instance
    '''
    
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    
    # Assign "label" --> LINE_NUMBER in a lookup table
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_name)
        
    #e.g. [[1, 2, 3], [4, 5, 6], [7, 8, 9]] --> [1, 2, 3, 4, 5, 6, 7, 8, 9] 
    dataset = files.flat_map(tf.data.TFRecordDataset)
    
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))

def parse_tfrecord(tfrecord, class_table, size):
    '''
    Parse operation for the encoded images and labels
    
    @param tfrecord: tf.data.dataset instance
    @param class_table: table with key-values pairs (classname-number)
    @param size:  size of the input image for the network
    
    '''
    # Read image-specific informations from the tfrecord file
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    #Decode image 
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    # Resize image to the desired size required by network
    x_train = tf.image.resize(x_train, (size, size))

    # Read the name of the class
    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    
    # Get the value (integer) for the class_text according to the class_table and convert to tf.float32
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    # Read groud truth values of the bounding boxes (these values are normed remember)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)
    
    #y_train shape: [Number of objects in image, (x1, y1, x2, y2, label)]

    # pad y_train with '0' so it has shape [100, 5]
    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
