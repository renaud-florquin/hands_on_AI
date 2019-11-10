
batch_size = 10

def create_fc_config(name, nbr_nodes, activation='relu', normalization=False, dropout=None):
    return {
      'name': name,
      'nbr_nodes': nbr_nodes,
      'normalization': normalization,
      'activation': activation,
      'dropout': dropout
    }                        

def learning_step(max_epochs, patience=10, freeze_until_layer_id=None, optimizer='Adagrad', optimizer_args=None):
    return {
      'freeze_until_layer_id': freeze_until_layer_id,
      'max_epochs': max_epochs,
      'patience': patience,
      'optimizer': {
          'class_id': optimizer,
          'args': {} if optimizer_args is None else optimizer_args
      }
    }



all_models = []

data_with_augmentation = {
  'repo': 'big',
  'categories': {0: 'fire', 1: 'no_fire', 2: 'start_fire'},
  'resolution': (224, 224),
  'batch_size': batch_size,
  'train_generator': {
    'with_data_augmentation': True,
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.1,
  }
}

data_without_augmentation = {
  'categories': {0: 'fire', 1: 'no_fire', 2: 'start_fire'},
  'resolution': (224, 224),
  'batch_size': batch_size,
  'train_generator': {
    'with_data_augmentation': False
  }
}

####################################################################################
# VGG16

VGG16_AvgPooling_3 = {
  'base_model': 'VGG16',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [],
}

VGG16_AvgPooling_64_3 = {
  'base_model': 'VGG16',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [
    create_fc_config('FC_1', 64)
  ],
}

VGG16_2steps_block5_AvgPooling_3 = {
  'config_id': 'VGG16_2steps_block5_AvgPooling_3',
  'data': data_with_augmentation,
  'model': VGG16_AvgPooling_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='block5', optimizer='SGD', optimizer_args={'learning_rate': 0.001})
  ]
}

VGG16_2steps_block5_AvgPooling_64_3 = {
  'config_id': 'VGG16_2steps_block5_AvgPooling_64_3',
  'data': data_with_augmentation,
  'model': VGG16_AvgPooling_64_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='block5', optimizer_args={'learning_rate': 0.001})
  ]
}

vgg16_models = [
   VGG16_2steps_block5_AvgPooling_3,
   VGG16_2steps_block5_AvgPooling_64_3
]

all_models.extend(vgg16_models)

####################################################################################
# Xception based models

Xception_AvgPooling_3 = {
  'base_model': 'Xception',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [],
}

Xception_AdaptivePooling_512_3 = {
  'base_model': 'Xception',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.3)
  ],
}

Xception_AvgPooling_128_64_3 = {
  'base_model': 'Xception',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [
    create_fc_config('FC_1', 128),
    create_fc_config('FC_2', 64)
  ],
}

Xception_2steps_block14_AvgPooling_3 = {
  'config_id': 'Xception_2steps_block14_AvgPooling_3',
  'data': data_with_augmentation,
  'model': Xception_AvgPooling_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='block14', optimizer='SGD', optimizer_args={'learning_rate': 0.001}),
  ]
}

Xception_2steps_block8_AdaptivePooling_512_3_with_norm = {
  'config_id': 'Xception_2steps_block8_AdaptivePooling_512_3_with_norm',
  'data': data_with_augmentation,
  'model': Xception_AdaptivePooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='block8', optimizer='SGD', optimizer_args={'learning_rate': 0.00001}),
  ]
}


Xception_2steps_block14_AvgPooling_128_64_3 = {
  'config_id': 'Xception_2steps_block14_AvgPooling_128_64_3',
  'data': data_with_augmentation,
  'model': Xception_AvgPooling_128_64_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='block14', optimizer='SGD', optimizer_args={'learning_rate': 0.001}),
  ]
}

Xception_4steps_block13_11_9_AvgPooling_128_64_3 = {
  'config_id': 'Xception_4steps_block13_11_9_AvgPooling_128_64_3',
  'data': data_with_augmentation,
  'model': Xception_AvgPooling_128_64_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='block13', optimizer='SGD', optimizer_args={'learning_rate': 0.001}),
    learning_step(max_epochs=100, freeze_until_layer_id='block11', optimizer='SGD', optimizer_args={'learning_rate': 0.0005}),
    learning_step(max_epochs=100, freeze_until_layer_id='block9', optimizer='SGD', optimizer_args={'learning_rate': 0.0001}),
  ]
}

xception_models = [
   Xception_2steps_block14_AvgPooling_3,
   Xception_2steps_block8_AdaptivePooling_512_3_with_norm,
   Xception_2steps_block14_AvgPooling_128_64_3,
   Xception_4steps_block13_11_9_AvgPooling_128_64_3
]

all_models.extend(xception_models)

####################################################################################
# ResNet50V2

ResNet50V2_AvgPooling_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [],
}

ResNet50V2_AvgPooling_512_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [
    create_fc_config('FC_1', 512)
  ],
}

ResNet50V2_2steps_conv5_AvgPooling_3 = {
  'config_id': 'ResNet50V2_2steps_conv5_AvgPooling_3',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='conv5', optimizer='SGD', optimizer_args={'learning_rate': 0.0001}),
  ]
}

ResNet50V2_2steps_conv5_AvgPooling_512_3 = {
  'config_id': 'ResNet50V2_2steps_conv5_AvgPooling_512_3',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='conv5', optimizer='SGD', optimizer_args={'learning_rate': 0.0001}),
  ]
}

resnet_models = [
   ResNet50V2_2steps_conv5_AvgPooling_3,
   ResNet50V2_2steps_conv5_AvgPooling_512_3
]

all_models.extend(resnet_models)


####################################################################################
# DenseNet121

DenseNet121_AvgPooling_3 = {
  'base_model': 'DenseNet121',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [],
}

DenseNet121_AvgPooling_512_3 = {
  'base_model': 'DenseNet121',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': False,
  'base_model_output_dropout': None,
  'classifier_topology': [
    create_fc_config('FC_1', 512)
  ],
}

DenseNet121_2steps_block5_28_AvgPooling_3 = {
  'config_id': 'DenseNet121_2steps_block5_28_AvgPooling_3',
  'data': data_with_augmentation,
  'model': DenseNet121_AvgPooling_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='conv5', optimizer='SGD', optimizer_args={'learning_rate': 0.0001}),
  ]
}

DenseNet121_2steps_block5_28_AvgPooling_512_3 = {
  'config_id': 'DenseNet121_2steps_block5_28_AvgPooling_512_3',
  'data': data_with_augmentation,
  'model': DenseNet121_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adagrad'),
    learning_step(max_epochs=100, freeze_until_layer_id='conv5', optimizer='SGD', optimizer_args={'learning_rate': 0.0001}),
  ]
}

densenet_models = [
   DenseNet121_2steps_block5_28_AvgPooling_3,
   DenseNet121_2steps_block5_28_AvgPooling_512_3
]

all_models.extend(resnet_models)

####################################################################################
# Experiment using REsNet
"""
ResNet50V2_AvgPooling_512_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2)
  ],
}

ResNet50V2_A_SGD_1 = {
  'config_id': 'ResNet50V2_A_SGD_1',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='SGD', optimizer_args={'learning_rate': 0.001})
  ]
}

ResNet50V2_A_SGD_2 = {
  'config_id': 'ResNet50V2_A_SGD_2',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='SGD', optimizer_args={'learning_rate': 0.0001})
  ]
}

ResNet50V2_A_RMSprop_1 = {
  'config_id': 'ResNet50V2_A_RMSprop_1',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='RMSprop', optimizer_args={'learning_rate': 0.001})
  ]
}

ResNet50V2_A_RMSprop_2 = {
  'config_id': 'ResNet50V2_A_RMSprop_2',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='RMSprop', optimizer_args={'learning_rate': 0.0001})
  ]
}

ResNet50V2_A_Adadelta_1 = {
  'config_id': 'ResNet50V2_A_Adadelta_1',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}

ResNet50V2_A_Adadelta_2 = {
  'config_id': 'ResNet50V2_A_Adadelta_2',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.0001})
  ]
}

ResNet50V2_B_SGD_1 = {
  'config_id': 'ResNet50V2_B_SGD_1',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='SGD', optimizer_args={'learning_rate': 0.001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv3', optimizer='SGD', optimizer_args={'learning_rate': 0.0001})
  ]
}

ResNet50V2_B_SGD_2 = {
  'config_id': 'ResNet50V2_B_SGD_2',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='SGD', optimizer_args={'learning_rate': 0.0001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv3', optimizer='SGD', optimizer_args={'learning_rate': 0.00001})
  ]
}

ResNet50V2_B_RMSprop_1 = {
  'config_id': 'ResNet50V2_B_RMSprop_1',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='RMSprop', optimizer_args={'learning_rate': 0.001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv3', optimizer='RMSprop', optimizer_args={'learning_rate': 0.0001})
  ]
}

ResNet50V2_B_RMSprop_2 = {
  'config_id': 'ResNet50V2_B_RMSprop_2',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='RMSprop', optimizer_args={'learning_rate': 0.0001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv3', optimizer='RMSprop', optimizer_args={'learning_rate': 0.00001})
  ]
}

ResNet50V2_B_Adadelta_1 = {
  'config_id': 'ResNet50V2_B_Adadelta_1',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv3', optimizer='Adadelta', optimizer_args={'learning_rate': 0.0001})
  ]
}

ResNet50V2_B_Adadelta_2 = {
  'config_id': 'ResNet50V2_B_Adadelta_2',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='Adadelta', optimizer_args={'learning_rate': 0.0001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv3', optimizer='Adadelta', optimizer_args={'learning_rate': 0.00001})
  ]
}

ResNet50V2_C_RMSprop_1 = {
  'config_id': 'ResNet50V2_C_RMSprop_1',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='start_classifier', optimizer='RMSprop', optimizer_args={'learning_rate': 0.0001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv4', optimizer='RMSprop', optimizer_args={'learning_rate': 0.00001}),
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='RMSprop', optimizer_args={'learning_rate': 0.000001})
  ]
}


resnet_optim_models = [
   ResNet50V2_A_SGD_1,
   ResNet50V2_A_SGD_2,
   ResNet50V2_A_RMSprop_1,
   ResNet50V2_A_RMSprop_2,
   ResNet50V2_A_Adadelta_1,
   ResNet50V2_A_Adadelta_2,
   ResNet50V2_B_SGD_1,
   ResNet50V2_B_SGD_2,
   ResNet50V2_B_RMSprop_1,
   ResNet50V2_B_RMSprop_2,
   ResNet50V2_B_Adadelta_1,
   ResNet50V2_B_Adadelta_2,
   ResNet50V2_C_RMSprop_1,
]
"""

ResNet50V2_AdaptivePooling_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
  ],
}


ResNet50V2_AdaptivePooling_512_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2)
  ],
}

ResNet50V2_AdaptivePooling_512_64_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2),
    create_fc_config('FC_2', 64, normalization=True, dropout=0.2)
  ],
}

ResNet50V2_AvgPooling_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
  ],
}


ResNet50V2_AvgPooling_512_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2)
  ],
}

ResNet50V2_AvgPooling_512_64_3 = {
  'base_model': 'ResNet50V2',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2),
    create_fc_config('FC_2', 64, normalization=True, dropout=0.2)
  ],
}

ResNet50V2_AvgPooling_3_Adadelta_0 = {
  'config_id': 'ResNet50V2_AvgPooling_3_Adadelta_0',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}

ResNet50V2_AvgPooling_512_3_Adadelta_0 = {
  'config_id': 'ResNet50V2_AvgPooling_512_3_Adadelta_0',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}

ResNet50V2_AvgPooling_512_64_3_Adadelta_0 = {
  'config_id': 'ResNet50V2_AvgPooling_512_64_3_Adadelta_0',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_64_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}


ResNet50V2_AdaptivePooling_3_Adadelta_0 = {
  'config_id': 'ResNet50V2_AdaptivePooling_3_Adadelta_0',
  'data': data_with_augmentation,
  'model': ResNet50V2_AdaptivePooling_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}

ResNet50V2_AdaptivePooling_512_3_Adadelta_0 = {
  'config_id': 'ResNet50V2_AdaptivePooling_512_3_Adadelta_0',
  'data': data_with_augmentation,
  'model': ResNet50V2_AdaptivePooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}

ResNet50V2_AdaptivePooling_512_64_3_Adadelta_0 = {
  'config_id': 'ResNet50V2_AdaptivePooling_512_64_3_Adadelta_0',
  'data': data_with_augmentation,
  'model': ResNet50V2_AdaptivePooling_512_64_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}


Xception_AdaptivePooling_512_3 = {
  'base_model': 'Xception',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2)
  ],
}

Xception_AdaptivePooling_512_3_Adadelta_0_block1 = {
  'config_id': 'Xception_AdaptivePooling_512_3_Adadelta_0_block1',
  'data': data_with_augmentation,
  'model': Xception_AdaptivePooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='block1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}

Xception_AdaptivePooling_512_3_Adadelta_0_block8 = {
  'config_id': 'Xception_AdaptivePooling_512_3_Adadelta_0_block8',
  'data': data_with_augmentation,
  'model': Xception_AdaptivePooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='block8', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}

DenseNet121_AdaptivePooling_512_3 = {
  'base_model': 'DenseNet121',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2)
  ],
}

DenseNet121_AdaptivePooling_512_3_Adadelta_0_conv5 = {
  'config_id': 'DenseNet121_AdaptivePooling_512_3_Adadelta_0_conv5',
  'data': data_with_augmentation,
  'model': DenseNet121_AdaptivePooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='conv5', optimizer='Adadelta', optimizer_args={'learning_rate': 0.01})
  ]
}


exp_models = [
   ResNet50V2_AvgPooling_3_Adadelta_0,
   ResNet50V2_AvgPooling_512_3_Adadelta_0,
   ResNet50V2_AvgPooling_512_64_3_Adadelta_0,
   ResNet50V2_AdaptivePooling_3_Adadelta_0,
   ResNet50V2_AdaptivePooling_512_3_Adadelta_0,
   ResNet50V2_AdaptivePooling_512_64_3_Adadelta_0,
   Xception_AdaptivePooling_512_3_Adadelta_0_block1,
   Xception_AdaptivePooling_512_3_Adadelta_0_block8,
   DenseNet121_AdaptivePooling_512_3_Adadelta_0_conv5,
]



Xception_AdaptivePooling_512_64_3 = {
  'base_model': 'Xception',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2),
    create_fc_config('FC_2', 64, normalization=True, dropout=0.2)
  ],
}

Xception_AdaptivePooling_512_64_3_using_sigmoid = {
  'base_model': 'Xception',
  'link_to_classifier': 'global_adaptive_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2, activation='sigmoid'),
    create_fc_config('FC_2', 64, normalization=True, dropout=0.2, activation='sigmoid')
  ],
}

Xception_AvgPooling_512_64_3 = {
  'base_model': 'Xception',
  'link_to_classifier': 'global_average_pooling2D',
  'base_model_output_normalization': True,
  'base_model_output_dropout': 0.2,
  'classifier_topology': [
    create_fc_config('FC_1', 512, normalization=True, dropout=0.2),
    create_fc_config('FC_2', 64, normalization=True, dropout=0.2)
  ],
}

Xception_AdaptivePooling_512_3_Adadelta_0_block1 = {
  'config_id': 'Xception_AdaptivePooling_512_3_Adadelta_0_block1',
  'data': data_with_augmentation,
  'model': Xception_AdaptivePooling_512_3,
  'training_steps': [
    learning_step(max_epochs=100, freeze_until_layer_id='block1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}

Xception_AvgPooling_512_64_3_Adadelta_0_block1 = {
  'config_id': 'Xception_AvgPooling_512_64_3_Adadelta_0_block1',
  'data': data_with_augmentation,
  'model': Xception_AvgPooling_512_64_3,
  'training_steps': [
    learning_step(max_epochs=200, freeze_until_layer_id='block1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}

Xception_AdaptivePooling_512_64_3_Adadelta_0_block1 = {
  'config_id': 'Xception_AdaptivePooling_512_64_3_Adadelta_0_block1',
  'data': data_with_augmentation,
  'model': Xception_AdaptivePooling_512_64_3,
  'training_steps': [
    learning_step(max_epochs=200, freeze_until_layer_id='block1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}

Xception_AdaptivePooling_512_64_3_Adadelta_0_block1_2steps = {
  'config_id': 'Xception_AdaptivePooling_512_64_3_Adadelta_0_block1_2steps',
  'data': data_with_augmentation,
  'model': Xception_AdaptivePooling_512_64_3,
  'training_steps': [
    learning_step(max_epochs=20, freeze_until_layer_id='start_classifier', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001}),
    learning_step(max_epochs=200, freeze_until_layer_id='block1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}


Xception_AdaptivePooling_512_64_3_Adadelta_0_block1_using_sigmoid = {
  'config_id': 'Xception_AdaptivePooling_512_64_3_Adadelta_0_block1_using_sigmoid',
  'data': data_with_augmentation,
  'model': Xception_AdaptivePooling_512_64_3_using_sigmoid,
  'training_steps': [
    learning_step(max_epochs=200, freeze_until_layer_id='block1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}


ResNet50V2_AvgPooling_512_64_3_Adadelta_0_low_LR = {
  'config_id': 'ResNet50V2_AvgPooling_512_64_3_Adadelta_0_low_LR',
  'data': data_with_augmentation,
  'model': ResNet50V2_AvgPooling_512_64_3,
  'training_steps': [
    learning_step(max_epochs=200, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}
ResNet50V2_AdaptivePooling_512_64_3_Adadelta_0_low_LR = {
  'config_id': 'ResNet50V2_AdaptivePooling_512_64_3_Adadelta_0_low_LR',
  'data': data_with_augmentation,
  'model': ResNet50V2_AdaptivePooling_512_64_3,
  'training_steps': [
    learning_step(max_epochs=200, freeze_until_layer_id='conv1', optimizer='Adadelta', optimizer_args={'learning_rate': 0.001})
  ]
}



exp2_models = [
   Xception_AdaptivePooling_512_3_Adadelta_0_block1,
   Xception_AvgPooling_512_64_3_Adadelta_0_block1,
   Xception_AdaptivePooling_512_64_3_Adadelta_0_block1,
   Xception_AdaptivePooling_512_64_3_Adadelta_0_block1_2steps,
   Xception_AdaptivePooling_512_64_3_Adadelta_0_block1_using_sigmoid,
   ResNet50V2_AvgPooling_512_64_3_Adadelta_0,
   ResNet50V2_AdaptivePooling_512_64_3_Adadelta_0,
   ResNet50V2_AvgPooling_512_64_3_Adadelta_0_low_LR,
   ResNet50V2_AdaptivePooling_512_64_3_Adadelta_0_low_LR
]
