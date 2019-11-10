### part of tl_train

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tl_utils

def create_generators(data_hyper_params, from_dir=None):
    """ Create a train and validation generators based on root dir (from_dir) and hyper_params. 
    :param data_hyper_params: the hyper parameters (mainly for resolution and data augmentation)
    :param from_dir: the directory with images (structure: train/<classes> + validation/<classes>)
    :returns: the train and validation generators 
    """
    from_dir = from_dir if from_dir is not None else data_hyper_params['repo']
    train_dir = os.path.join(from_dir, 'train')
    validation_dir = os.path.join(from_dir, 'validation')  

    if data_hyper_params['train_generator']['with_data_augmentation']:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1./255,
          rotation_range=data_hyper_params['train_generator']['rotation_range'],
          width_shift_range=data_hyper_params['train_generator']['width_shift_range'],
          height_shift_range=data_hyper_params['train_generator']['height_shift_range'],
          shear_range=data_hyper_params['train_generator']['shear_range'],
          zoom_range=data_hyper_params['train_generator']['zoom_range'],
          horizontal_flip=data_hyper_params['train_generator']['horizontal_flip'],
          fill_mode='nearest')
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1./255)

    test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )

    # Flow training images in batches using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                       batch_size=data_hyper_params['batch_size'],
                                                       class_mode='categorical',
                                                       target_size = data_hyper_params['resolution'])

    # Flow validation images in batches using test_datagen generator
    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=data_hyper_params['batch_size'],
                                                            class_mode  = 'categorical',
                                                            target_size = data_hyper_params['resolution'])
    return train_generator, validation_generator


def build_model(model_hyper_params, nbr_categories, input_resolution):
    """ Based on hyper_params create a model using transfer learning (pre-defined model + classifier layers).
    :param model_hyper_params: the hyper parameters
    :param nbr_categories: number of output categories (classification)
    :param input_resolution: input image resolution; tuple of (rows, cols)
    :returns: the created model
    """

    # base model using pre-trained model
    base_model_class = getattr(tf.keras.applications, model_hyper_params['base_model'])
    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=(input_resolution[0], input_resolution[1], 3))

    x = base_model.output
    
    
    if model_hyper_params['link_to_classifier'] == 'global_average_pooling2D':
        x = tf.keras.layers.GlobalAveragePooling2D(name='start_classifier')(x)
    elif model_hyper_params['link_to_classifier'] == 'global_max_pooling2D':
        x = tf.keras.layers.GlobalMaxPool2D(name='start_classifier')(x)
    elif model_hyper_params['link_to_classifier'] == 'global_adaptive_pooling2D':
        x1 = tf.keras.layers.GlobalAveragePooling2D(name='start_classifier_average')(x)
        x2 = tf.keras.layers.GlobalMaxPool2D(name='start_classifier_max')(x)
        concat = tf.keras.layers.Concatenate(name='start_classifier')
        x = concat([x1, x2])    
    else:
        x = tf.keras.layers.Flatten(name='start_classifier')(x)

    if model_hyper_params['base_model_output_normalization']:
        x = tf.keras.layers.BatchNormalization()(x)

    if model_hyper_params['base_model_output_dropout']:
        x = tf.keras.layers.Dropout(rate=model_hyper_params['base_model_output_dropout'])(x)

    for layer_topology in model_hyper_params['classifier_topology']:
        if layer_topology['normalization']:
            x = tf.keras.layers.Dense(layer_topology['nbr_nodes'], name=layer_topology['name'], use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(layer_topology['activation'])(x)
        else:
            x = tf.keras.layers.Dense(layer_topology['nbr_nodes'], activation=layer_topology['activation'], name=layer_topology['name'])(x)
        if layer_topology['dropout']:
            x = tf.keras.layers.Dropout(rate=layer_topology['dropout'])(x)

    predictions = tf.keras.layers.Dense(nbr_categories, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=predictions)

    return model
                        
def freeze_model_until_layer(model, layer_id=None):
    trainable = False
    for l in model.layers:
        if not trainable and layer_id is not None and l.name.startswith(layer_id):
            trainable = True
        l.trainable = trainable

def create_optimizer(optimizer_hyper_params):
    optimizer_class = getattr(tf.keras.optimizers, optimizer_hyper_params['class_id'])
    return optimizer_class(**optimizer_hyper_params['args'])

def train(model, training_hyper_params, train_generator, validation_generator, output_dir):
    results = {
        'steps': [],
        'acc': [],
        'val_acc': [],
        'loss': [],
        'val_loss': []
    }
    epochs = 0
    t_begin = time.time()
    for step in training_hyper_params:
        step_result = {}
        if step['freeze_until_layer_id'] is not None:
            freeze_model_until_layer(model, layer_id=step['freeze_until_layer_id'])
        model.compile(loss='categorical_crossentropy', optimizer=create_optimizer(step['optimizer']), metrics=['acc'])
        # dump model
        summary = []
        model.summary(print_fn=lambda info: summary.append(info))
        step_result['model_summary'] = ''.join(summary)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=step['patience'], verbose=0, mode='min')
        filepath= output_dir + '/model-{epoch:03d}.hdf5'
        model_saver = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
        history = tf.keras.callbacks.History()
        t_step_begin = time.time()
        model.fit_generator(
          train_generator,
          epochs=epochs+step['max_epochs'],
          validation_data=validation_generator,
          callbacks=[early_stopping, model_saver, history],
          initial_epoch=epochs)
        t_step_end = time.time()
        step_result['duration'] = t_step_end - t_step_begin
        step_result['first_epoch'] = history.epoch[0]
        step_result['last_epoch'] = history.epoch[-1]
        step_result['nbr_epochs'] = step_result['last_epoch'] - step_result['first_epoch'] + 1
        epochs = epochs + step_result['nbr_epochs']
        results['steps'].append(step_result)
        results['acc'].extend(map(float, history.history['acc']))
        results['val_acc'].extend(map(float, history.history['val_acc']))
        results['loss'].extend(map(float, history.history['loss']))
        results['val_loss'].extend(map(float, history.history['val_loss']))
    t_end = time.time()
    results['total_duration'] = t_end - t_begin
    return results


def train_analyse(hyper_params, result_dir, from_dir=None):
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, hyper_params['config_id'])
    os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'hyper_parameters.json'), 'w') as f:
        json.dump(hyper_params, f)
    train_generator, validation_generator = create_generators(hyper_params['data'], from_dir)
    model = build_model(hyper_params['model'], len(hyper_params['data']['categories']), hyper_params['data']['resolution'])
    results = train(model, hyper_params['training_steps'], train_generator, validation_generator, str(output_dir))
    tl_utils.analyse_result(hyper_params, results, output_dir, validation_generator)
    return results

def campaign(models, result_dir, from_dir=None):
    model_config_ids = [m['config_id'] for m in models]
    nbr_models = len(model_config_ids)
    assert nbr_models == len(set(model_config_ids)), 'The config_id must be unique'
    summary_df = pd.DataFrame(columns=['nbr_steps','nbr_epochs', 'total_duration', 'best_epoch', 'optimal_model', 'best_acc', 'best_val_acc', 'best_loss', 'best_val_loss'])
    for i, model_param in enumerate(models):
        print('Start model [{}/{}]: {}'.format(i+1, nbr_models, model_param['config_id']))
        results = train_analyse(model_param, result_dir, from_dir)
        summary_df.loc[model_param['config_id']] = [
             len(results['steps']),
             len(results['acc']),
             results['total_duration'],
             results['optimal_epoch'],
             results['optimal_model'],
             results['best_acc'],
             results['best_val_acc'],
             results['best_loss'],
             results['best_val_loss']           
        ]
        tf.keras.backend.clear_session()
    summary_df.to_csv(os.path.join(result_dir, 'summary.csv'))
    summary_df.to_excel(os.path.join(result_dir, 'summary.xlsx'))