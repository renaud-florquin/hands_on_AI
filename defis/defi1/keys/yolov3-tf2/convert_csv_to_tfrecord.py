"""Convert a set of images and a CSV annotation file to TFRecord for object_detection.

Example usage:
    python convert_csv_to_tfrecord.py \
        --img_dir=keys_and_background \
        --csv_filename=keys_and_background/annotations.csv \
        --output_path=tfrecord_output
"""

import os
import io
import hashlib
import PIL
import pandas as pd
import tensorflow as tf


from absl import flags
from absl import logging
from absl import app
FLAGS = flags.FLAGS


flags.DEFINE_string('img_dir', '', 'Root directory to images dataset.')
flags.DEFINE_string('csv_filename', 'train', 'CSV annotations of the images.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')

SAMPLES_PER_FILES = 200


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_output_filename(output_dir, name, idx):
    return os.path.join(output_dir, '{}_{:03}.tfrecord'.format(name, idx))


def add_to_tfrecord(data, image_dir, tfrecord_writer):

    img_filename = os.path.join(image_dir, data.filename)
    with tf.io.gfile.GFile(img_filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(img_filename)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    sha256 = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    
    xmin.append(float(data.xmin) / width)
    ymin.append(float(data.ymin) / height)
    xmax.append(float(data.xmax) / width)
    ymax.append(float(data.ymax) / height)
    classes_text.append(data.class_name.encode('utf8'))
    classes.append(data.class_id)
    truncated.append(int(0))
    poses.append(''.encode('utf8'))
    difficult_obj.append(int(0))  

    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(data.filename.encode('utf8')),
            'image/source_id': bytes_feature(data.filename.encode('utf8')),
            'image/key/sha256': bytes_feature(sha256.encode('utf8')),
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/text': bytes_list_feature(classes_text),
            'image/object/class/label': int64_list_feature(classes),
            'image/object/bbox/difficult': int64_list_feature(difficult_obj),
            'image/object/bbox/truncated': int64_list_feature(truncated),
            'image/object/view': bytes_list_feature(poses),
        }))
    tfrecord_writer.write(example.SerializeToString())


def convert_csv_to_tfrecord(annotation_filename, image_dir, output_path):
    data_csv = pd.read_csv(annotation_filename)
    splitted_dfs = [data_csv.loc[i:i+SAMPLES_PER_FILES-1,:] for i in range(0, len(data_csv), SAMPLES_PER_FILES)]    
    for tf_idx, df in enumerate(splitted_dfs):
        tf_filename = get_output_filename(output_path, 'keys', tf_idx)
        with tf.io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for i in df.index:
                add_to_tfrecord(df.loc[i], image_dir, tfrecord_writer)


def main(_):
    convert_csv_to_tfrecord(FLAGS.csv_filename, FLAGS.img_dir, FLAGS.output_path)

if __name__ == '__main__':
  app.run(main)

