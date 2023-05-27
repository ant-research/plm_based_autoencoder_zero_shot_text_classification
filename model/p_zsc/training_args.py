class TrainingArgs:
    '''
    a Args class that maintain all arguments for model training
    '''
    output_path = ""
    fulltrain_batch_size = 16
    pretrain_batch_size = 8
    selftrain_batch_size = 128

    pre_train_epochs = 1
    self_train_epochs = 3
    full_train_epochs = 3

    pre_train_eval_steps = -1
    self_train_eval_steps = -1
    full_train_eval_steps = -1

    accumulation_steps = 1
    selftrain_accumulation_steps=1
    pre_train_training_lr = 2e-5
    full_train_training_lr = 2e-5
    self_train_training_lr = 1e-6

    warmup_ratio = 0.1
    eval_batch_size = 32
    pretrain_lr_scheduler = "linear"
    selftrain_lr_scheduler = "linear"

    seed = 2021
    weight_decay = 0.1

    self_train_update_interval = 50
    self_train_early_stop = False

    override = False
    pre_train_one_hot=True
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
