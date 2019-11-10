### part of tl_utils
import os
import json
#import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def display_images_with_class(bad_classified_images, img_columns = 4):
    """ Display several images with current and expected class.
    :param bad_classified_images: sequence of tuples (filename, current class, expected class)
    :param img_columns: number of columns to display
    """
    fig = plt.figure()
    img_rows = (len(bad_classified_images) // img_columns) + 1
    for i, (file_name, current_class, expected_class) in enumerate(bad_classified_images):
        ax = fig.add_subplot(img_rows, img_columns, 1 + i)
        img = tf.keras.preprocessing.image.load_img(file_name, target_size=(224,224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x /= 255.0
        plt.imshow(x)
        plt.title('{}/{}'.format(current_class, expected_class))
        plt.xticks([]) 
        plt.yticks([])

    plt.show()


def display_samples(generator, category_map, img_rows = 2, img_columns = 4):
    """ Display several images (img_rows x img_columns) with the category as title using the generator.
    :param generator: the generator to extract a first set of images
    :param category_map: the category map (index=>category id)
    :param img_rows: number of row to  display
    :param img_columns: number of columns to display
    """
    x_step, y_step = generator.next()
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

def display_metrics(results, metric_id, output_file, configuration_id=None, opt_fct=None):
    """ Display metrics based on training history and save the figure to file.
    :param history: the training history
    :param metric_id: the metric to display (i.e. 'acc' or 'loss'
    :param output_file: the output file to save the figure
    """
    metric_id1 = metric_id
    metric_id2 = 'val_{}'.format(metric_id)
    metric = results[metric_id1]
    val_metric = results[metric_id2]
    epochs   = range(1, len(metric)+1)

    plt.plot(epochs, metric, 'darkcyan', label=metric_id1)
    plt.plot(epochs, val_metric, 'forestgreen', label=metric_id2)
    if opt_fct:
        optimal_val = opt_fct(val_metric)
        optimal_epoch = val_metric.index(optimal_val) + 1
        plt.plot([optimal_epoch], [optimal_val], 'ro')
        plt.title('Training/Validation for {} [best epoch: {}]'.format(metric_id, optimal_epoch))
    else:
        plt.title('Training/Validation for {}'.format(metric_id))

    plt.ylabel(metric_id)
    plt.xlabel('Epochs')
    if configuration_id:
        plt.suptitle(configuration_id)
    plt.legend()
    for step in results['steps'][:-1]:
        plt.axvline(x=step['last_epoch']+1, color='grey', linestyle='--')

    plt.savefig(output_file)
    plt.close()  

def display_confusion_matrix(y_true, y_pred, output_file):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, columns = np.unique(y_true), index = np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sn.set(font_scale=1.0)
    sns_plot = sn.heatmap(df_cm, cmap="Blues", annot=True)
    fig = sns_plot.get_figure()
    fig.savefig(output_file)
    plt.close()

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


def scan_and_classify_image(model, from_dir, categories, resolution=(224, 224)):
    """
    :param model: the model to use to analyse the picture
    :param from_dir: the source of images
    :param categories: a dictionary of categories
    :param resolution: image resolution
    """
    bad_classified = []

    for root, dirs, files in os.walk(from_dir):
            for file in files:
                if file[-4:] in ['.jpg', '.png', 'jpeg']:
                    file_name = os.path.join(root, file)
                    expected_category = root.split('/')[-1]
                    img_class, class_probabilities = classify_image(file_name, model, resolution=resolution)
                    selected_category = categories[img_class]
                    print('{img_class:12} - {file_name} ({class_probabilities})'.format(file_name=file_name, img_class=selected_category, class_probabilities=class_probabilities))
                    if expected_category != selected_category:
                        bad_classified.append((file_name, selected_category, expected_category))

    return bad_classified


def analyse_result(hyper_params, results, output_dir, validation_generator):
    display_metrics(results, 'acc', os.path.join(output_dir, 'acc.png'), hyper_params['config_id'], opt_fct=max)
    display_metrics(results, 'loss', os.path.join(output_dir, 'loss.png'), hyper_params['config_id'], opt_fct=min)

    results['best_acc'] =  max(results['acc'])
    results['best_val_acc'] = max(results['val_acc'])
    results['best_loss'] = min(results['loss'])
    results['best_val_loss'] = min(results['val_loss'])

    results['optimal_epoch'] = optimal_epoch = results['val_loss'].index(min(results['val_loss'])) + 1
    results['optimal_model'] = 'model-{epoch:03d}.hdf5'.format(epoch=optimal_epoch)
    best_model_file = os.path.join(output_dir, results['optimal_model'])
    best_model = tf.keras.models.load_model(best_model_file)
    Y_pred = best_model.predict_generator(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes
    category_dict = hyper_params['data']['categories']
    y_true_label = [category_dict[y] for y in y_true]
    y_pred_label = [category_dict[y] for y in y_pred]
    #display_confusion_matrix(y_true_label, y_pred_label, os.path.join(output_dir, 'confusion_matrix.png'))
    target_names = [category_dict[i] for i in range(len(category_dict))]
    results['classification_report'] = classification_report(y_true, y_pred, target_names=target_names)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    return results
