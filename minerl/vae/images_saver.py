import minerl
import os
import numpy as np
import time
from matplotlib import pyplot as plt

"""
This file is used to read the data from the minerl DataPipeline and save the images into numpy matrices on disk.
I decided to do this because minerl DataPipeline is unstable and limited a lot the way I intend to process the images.
"""

def load_generators():
    datasets_path = '../minerl-dataset'
    generators = []
    names = []

    for f in os.listdir(datasets_path):
        if f.startswith('MineRL'):
            data = minerl.data.make(f, data_dir=datasets_path, num_workers=8)
            generators.append(data)
            names.append(f)
    return generators, names


def save_data(generator_idx, skip=3):
    generators, names = load_generators()
    generator = generators[generator_idx]
    name = 'images/'+names[generator_idx]
    print('Loading '+name)
    pos = 0
    data = np.empty((0, 64, 64, 3), dtype=np.uint8)
    for s, a, r, s_prime, d in generator.sarsd_iter(num_epochs=1, max_sequence_len=32):
        try:
            pov = s['pov']
            pov = pov[range(0, pov.shape[0], skip)]
            if pos + pov.shape[0] >= data.shape[0]:
                data.resize((data.shape[0] + pov.shape[0], data.shape[1], data.shape[2], data.shape[3]))
            data[pos: pos + pov.shape[0]] = pov
            pos += pov.shape[0]
        except:
            print('Caught')
    print('Saving '+name)
    np.save(file=name, arr=data)


if __name__ == "__main__":
    _, names = load_generators()
    for idx, name in enumerate(names):
        if name+'.npy' in os.listdir('images/'):
            print('Skipping ' + name)
            continue
        print('Processing '+name)
        save_data(generator_idx=idx, skip=4)
