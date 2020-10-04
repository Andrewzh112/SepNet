class Config:
    epochs = 1000
    batch_size = 64
    opimizer_params = {
        'lr': 1e-3,
        'betas': (0.5, 0.99),
        'weight_decay': 0.01
    }
    grad_norm_clip = 1.0
    num_workers = 4
    model_path = 'model_weights'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
