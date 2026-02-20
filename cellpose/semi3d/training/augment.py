import numpy as np


def simulate_z_dropout(mask_stack, dropout_prob=0.2, seed=0):
    rng = np.random.default_rng(seed)
    aug = mask_stack.copy()
    for z in range(len(aug)):
        if rng.random() < dropout_prob:
            aug[z] = 0
    return aug
