import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

default_hyper_params = {
  'config_id': 'Big_VGG16_no_freeze_Adagrad_512_128_64_flatten',
  'categories': {0: 'fire', 1: 'no_fire', 2: 'start_fire'},
  'epochs': 100,
  'batch_size': 50,
  'patience': 10,
  'resolution': (224, 224),
  'train_generator': {
     'with_data_augmentation': True,
     'rotation_range': 15,
     'width_shift_range': 0.1,
     'height_shift_range': 0.1,
     'shear_range': 0.1,
     'horizontal_flip': True,
     'zoom_range': 0.1,
  },
  'base_model': 'VGG16',
  'freeze_base_model': False,
  'trainable_layers': 0,                   # means trainable layers from [0:]
  'use_global_average_pooling2D': False,
  'optimizer': 'Adagrad',
  'classifier_topology': [
      {
          'nbr_nodes': 512,
          'activation': 'relu',
          'dropout': 0,
      },                        
      {
          'nbr_nodes': 128,
          'activation': 'relu',
          'dropout': 0,
      },                        
      {
          'nbr_nodes': 64,
          'activation': 'relu',
          'dropout': 0,
       }                        
  ],
}

def create_generators(from_dir, hyper_params):
    """ Create a train and validation generators based on root dir (from_dir) and hyper_params. 
    :param from_dir: the directory with images (structure: train/<classes> + validation/<classes>)
    :param hyper_params: the hyper parameters (mainly for resolution and data augmentation)
    :returns: the train and validation generators 
    """
    train_dir = os.path.join(from_dir, 'train')
    validation_dir = os.path.join(from_dir, 'validation')  

    if hyper_params['train_generator']['with_data_augmentation']:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1./255,
          rotation_range=hyper_params['train_generator']['rotation_range'],
          width_shift_range=hyper_params['train_generator']['width_shift_range'],
          height_shift_range=hyper_params['train_generator']['height_shift_range'],
          shear_range=hyper_params['train_generator']['shear_range'],
          zoom_range=hyper_params['train_generator']['zoom_range'],
          horizontal_flip=hyper_params['train_generator']['horizontal_flip'],
          fill_mode='nearest')
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1./255)

    test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )

    # Flow training images in batches using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                       batch_size=hyper_params['batch_size'],
                                                       class_mode='categorical',
                                                       target_size = hyper_params['resolution'])

    # Flow validation images in batches using test_datagen generator
    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=hyper_params['batch_size'],
                                                            class_mode  = 'categorical',
                                                            target_size = hyper_params['resolution'])
    return train_generator, validation_generator


def display_samples(generator, category_map, img_rows = 2, img_columns = 4):
    """ Display several images (img_rows x img_columns) with the category as title using the generator.
    :param generator: the generator to extract a first set of images
    :param category_map: the category map (index=>category id)
    :param img_rows: number of row to  display
    :param img_columns: number of columns to display
    """
    x_step, y_step = train_generator.next()
    amount = img_rows * img_columns
    assert x_step.shape[0] >= amount, 'Not enough images per iteration'
    fig = plt.figure()

    for i in range(amount):
        ax = fig.add_subplot(img_rows, img_columns, 1 + i)
        category_index = np.argmax(y_step[i])
        plt.imshow(x_step[i])
        plt.title(category_map[category_index])
        plt.xticks([]) 
        plt.yticks([])

    plt.show()

def build_model(hyper_params):
    """ Based on hyper_params create a model using transfer learning (pre-defined model + classifier layers).
    :param hyper_params: the hyper parameters
    :returns: the created model
    """

    # base model using pre-trained model
    base_model_class = getattr(tf.keras.applications, hyper_params['base_model'])
    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=(hyper_params['resolution'][0],hyper_params['resolution'][1],3))

    for layer in base_model.layers:
        layer.trainable = False

    if not hyper_params['freeze_base_model']:
        for layer in base_model.layers[hyper_params['trainable_layers']:]:
            layer.trainable = True  

    x = base_model.output
    if hyper_params['use_global_average_pooling2D']:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = tf.keras.layers.Flatten()(x)

    for layer_topology in hyper_params['classifier_topology']:
        x = tf.keras.layers.Dense(layer_topology['nbr_nodes'], activation=layer_topology['activation'])(x)
        if layer_topology['dropout']:
            x = tf.keras.layers.Dropout(rate=layer_topology['dropout'])(x)

    predictions = tf.keras.layers.Dense(len(hyper_params['categories']), activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=predictions)

    return model


def display_metrics(history, metric_id, output_file):
    """ Display metrics based on training history and save the figure to file.
    :param history: the training history
    :param metric_id: the metric to display (i.e. 'acc' or 'loss'
    :param output_file: the output file to save the figure
    """
    metric = history.history[metric_id]
    val_metric = history.history['val_' + metric_id]
    epochs   = range(1, len(metric)+1)

    plt.plot(epochs, metric, 'b')
    plt.plot(epochs, val_metric, 'r')
    plt.title('Training/Validation for {}'.format(metric_id))

    plt.savefig(output_file)  

def classify_image(file_name, model, resolution=(224, 224)):
    """ Print the class of an image (defined by file_name) using a trained model.
    :param file_name: the image's file
    :param model: the trained model
    :param resolution: image resolution
    :returns: the category index and the class probabilities vector
    """
    img = tf.keras.preprocessing.image.load_img(file_name, target_size=resolution)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    batch_class_probabilities = model.predict(images, batch_size=10)
    class_probabilities = batch_class_probabilities[0] 
    return np.argmax(class_probabilities), class_probabilities

def scan_and_classify_image(model_file_name, from_dir, hyper_params):
    """
    :param hyper_params: the hyper parameters
    """
    index_to_class = {0: 'Fire', 1: 'No Fire', 2: 'Start Fire'}
    model = tf.keras.models.load_model(model_file_name)

    for root, dirs, files in os.walk(from_dir):
            for file in files:
                if file[-4:] in ['.jpg', '.png', 'jpeg']:
                     img_class, class_probabilities = classify_image(os.path.join(root, file), model, resolution=hyper_params['resolution'])
                     print('{img_class:12} - {file_name} ({class_probabilities})'.format(file_name=file_name, img_class=hyper_params['categories'][img_class], class_probabilities=class_probabilities))

