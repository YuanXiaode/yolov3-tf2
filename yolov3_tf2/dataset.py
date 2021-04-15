import tensorflow as tf
from absl.flags import FLAGS
from absl import logging

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (None, num_boxes, (x1, y1, x2, y2, class, best_anchor))
    # anchor_idxs : (3)
    N = tf.shape(y_true)[0]

    # y_true_out: (None, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N): ## N
        for j in tf.range(tf.shape(y_true)[1]): ## num_boxes
            if tf.equal(y_true[i][j][2], 0):  ## 去除空的box，前面提过，若样本的box少于yolo_max_boxes个，就补零
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))  ## 找到最好的anchor, anchor_eq shape:(3,)
            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2  ## 角点->中心  shape:(2,)

                ## tf.where(anchor_eq) 返回anchor_eq为True的index,若有n个True，则返回的shape为(n,1)
                ## 其实这里anchor_eq只可能有一个True，所以返回的shape = (1,1)
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)

                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]) ## 保存了坐标和类别
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


## 这里维度搞了半天，只要前面调用过 train_dataset.batch(FLAGS.batch_size)，这里的y_train维度就是(None，num_boxes，5)
## 若没调用 train_dataset.batch， 维度就是 (num_boxes，5)
## 上面之所以多了一个维度，应该是map函数将一个batch的数据并行处理了
def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32
    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]  ## anchors 只有w,h  shape: (9,2)
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]  ## (None,num_boxes,2)

    ## (None,num_boxes,9,2)
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    ## 这里是将box和anchors移到中心重合后计算的IOU
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection) ## shape (None,num_boxes,9)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)  ## 从 9 个 anchors 中找到和 box 的 IOU 最大的那个
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)  ## shape:(None，num_boxes，6)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):  ## YunYang代码用的letterBox
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


def parse_tfrecord(tfrecord, class_table, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')  ## 稀疏表示转为正常矩阵，形式应该是这样["cat", "hot dog", "...",...]维度为（N,）
    labels = tf.cast(class_table.lookup(class_text), tf.float32)  ## 维度和class_text一样的,[15,52,-1,-1,-1...]
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)  ## 维度应该是 (N,5)

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)  ## 维度应该是 (FLAGS.yolo_max_boxes,5)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=416):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    ## tf.lookup.TextFileInitializer 用法：
    ## class_file 表示读取的文件
    ## tf.string, 0 表示key的类型（这里是类别名），0 表示基于后面的换行符（delimiter="\n"）来分割
    ## tf.int64, LINE_NUMBER 表示value的类型，LINE_NUMBER = -1 表示用从零开始的行号作为value

    ## 而 tf.lookup.StaticHashTable 就是一个字典而已，找不到该键，就返回-1
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    ## 获取符合 file_pattern 的所有file
    files = tf.data.Dataset.list_files(file_pattern)
    ## files 中的每个文件都调用tf.data.TFRecordDataset，并将结果放到一起（相当于在map之后，将多个对象合并成一个对象）
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/girl.png', 'rb').read(), channels=3)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0], ## ->(x,y,x,y,label)
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5  ## (8,5)，其中后五个是无效数据
    y_train = tf.convert_to_tensor(labels, tf.float32)

    # from_tensor_slices 切片操作，要求第一个维度是样本数量
    # 读取数据时，会在第一个维度切分，因此读取时的维度是 (size,size,3) 和 (8,5)
    x_train = tf.expand_dims(x_train, axis=0)  ## (1,size,size,3)
    y_train = tf.expand_dims(y_train, axis=0)  ## (1,8,5)


    ## 输出形式和 parse_tfrecord 函数一样
    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
