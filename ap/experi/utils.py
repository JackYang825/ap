"""Utility functions for experiments."""


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    )
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    )
    return train_dataset, eval_dataset














