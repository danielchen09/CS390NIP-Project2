DEFAULT_CONFIG = {
    'epochs': 15,
    'batch_size': 32,
    'augment': False,
    '64_layers': 1,
    '128_layers': 1,
    '256_layers': 1,
    'use_dropout': True,
    'dropout': 0.25,
    'kernel_initializer': 'glorot_uniform',
    'use_regularization': True,
    'regularization': 'l2',
    'lambda': 0.01,
    'linear_layers': [512],
    'auto_lr': False,
    'save': True
}


def edit_default_config(conf):
    config = dict(DEFAULT_CONFIG)
    for k in config:
        if k in conf:
            config[k] = conf[k]
    for k in conf:
        if k not in config:
            config[k] = conf[k]
    return config


def cifar100f_config():
    conf = {
        'epochs': 25,
        'augment': True,
        '64_layers': 2,
        '128_layers': 2,
        '256_layers': 2,
        'linear_layers': [1024, 512],
        'use_regularization': True,
        'lambda': 0.01,
        'auto_lr': True
    }
    return edit_default_config(conf)