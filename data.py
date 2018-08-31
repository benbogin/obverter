import pickle
import random
from PIL import Image
import numpy as np

colors = {'red': [1, 0, 0], 'blue': [0, 0, 1], 'green': [0, 1, 0], 'white': [1]*3, 'gray': [0.5]*3,
          'yellow': [1, 1, 0.1], 'cyan': [0, 1, 1], 'magenta': [1, 0, 1]}

object_types = ['box', 'sphere', 'cylinder', 'torus', 'ellipsoid']


def load_images_dict(data_n_samples):
    cache_filename = 'assets/cache.pkl'

    print("Looking for cache file %s" % cache_filename)
    try:
        images_cache = pickle.load(open(cache_filename, 'rb'))
        return images_cache
    except FileNotFoundError:
        print('No cache file, trying to create one...')
    except Exception as e:
        print('Error loading cache file', e)
        exit()

    images_cache = {}
    for color in colors:
        for object_type in object_types:
            for i in range(0, data_n_samples):
                path = 'assets/%s-%s-%d.png' % (color, object_type, i)
                images_cache[color, object_type, i] = np.array(
                    list(Image.open(path).getdata())).reshape((128, 128, 3))

    pickle.dump(images_cache, open('assets/cache.pkl', 'wb'))
    print("Saved cache file %s" % cache_filename)

    return images_cache


def pick_random_color(exclude=None):
    available_colors = list(colors.keys())
    if exclude is not None:
        available_colors.remove(exclude)

    return random.choice(available_colors)


def pick_random_object_type(exclude=None):
    available_object_types = list(object_types)
    if exclude is not None:
        available_object_types.remove(exclude)

    return random.choice(available_object_types)


def get_batches(images_cache, data_n_samples, n_batches=20, batch_size=50):
    batches = []

    n_same = int(0.25*batch_size)
    n_same_shape = int(0.3*batch_size)
    n_same_color = int(0.2*batch_size)
    n_random = batch_size - n_same_shape - n_same_color - n_same

    for ib in range(n_batches):
        pairs = []

        for i in range(n_same):
            object_type, color = pick_random_object_type(), pick_random_color()
            pairs.append(((object_type, color), (object_type, color)))

        for i in range(n_same_shape):
            object_type, color = pick_random_object_type(), pick_random_color()
            color2 = pick_random_color(exclude=color)
            pairs.append(((object_type, color), (object_type, color2)))

        for i in range(n_same_color):
            object_type, color = pick_random_object_type(), pick_random_color()
            object_type2 = pick_random_object_type(exclude=object_type)
            pairs.append(((object_type, color), (object_type2, color)))

        for i in range(n_random):
            object_type, color = pick_random_object_type(), pick_random_color()
            object_type2, color2 = pick_random_object_type(), pick_random_color()
            pairs.append(((object_type, color), (object_type2, color2)))

        input1 = []
        input2 = []
        labels = []
        descriptions = []

        for pair in pairs:
            max_i = data_n_samples
            (object_type1, color1), (object_type2, color2) = pair
            label = object_type1 == object_type2 and color1 == color2

            id1 = random.randint(0, max_i-1)
            img1 = images_cache[color1, object_type1, id1] / 256

            if label:
                available_ids = list(range(id1)) + list(range(id1+1, max_i))
                id2 = random.choice(available_ids)
            else:
                id2 = random.randint(0, max_i - 1)
            img2 = images_cache[color2, object_type2, id2] / 256

            input1.append(img1)
            input2.append(img2)
            labels.append(int(label))
            descriptions.append(((object_type1, color1), (object_type2, color2)))

        batches.append((np.array(input1), np.array(input2), labels, descriptions))

    return batches
