class Config:
    epochs = 100
    batch_size = 8
    opimizer_params = {
        'lr': 3e-4,
        'momentum': 0.5,
        'weight_decay': 0.1,
        'nesterov': True
    }
    grad_norm_clip = 1.0
    num_workers = 4
    model_path = 'model_weights/'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
