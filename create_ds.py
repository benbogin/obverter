import os
import random
import argparse

from vapory import *

from data import colors, object_types


class Torus(POVRayElement):
    """"""


def render_scene(filename, object_type, color, location, rotation):
    assert (object_type in object_types)
    assert (color in colors)

    color = colors[color]
    size = 2
    radius = size/2
    attributes = Texture(Pigment('color', color)), Finish('ambient', 0.7), 'rotate', (0, rotation, 0)
    if object_type == 'box':
        location.insert(1, size/2)
        obj = Box([x - size/2 for x in location], [x + size/2 for x in location], *attributes)
    if object_type == 'sphere':
        location.insert(1, radius)
        obj = Sphere(location, radius, *attributes)
    if object_type == 'torus':
        location.insert(1, radius/2)
        obj = Torus(radius, radius/2, 'translate', location, *attributes)
    if object_type == 'ellipsoid':
        location.insert(1, radius)
        obj = Sphere(location, radius, 'scale', (0.75, 0.45, 1.5), *attributes)
    if object_type == 'cylinder':
        location.insert(1, 0)
        location2 = list(location)
        location2[1] = size*2
        obj = Cylinder(location, location2, radius, *attributes)

    camera = Camera('location', [0, 8, 7], 'look_at', [0, 0, 0])
    light = LightSource([0, 10, 0], 'color', [1, 1, 1])

    chessboard = Plane([0, 1, 0], 0, 'hollow',
                       Texture(Pigment('checker',
                                       'color', [.47, .6, .74],
                                       'color', [.34, 0.48, 0.6]),
                               'scale', 4), Finish('ambient', 0.5))

    scene = Scene(camera, objects=[light, obj, chessboard])
    scene.render(filename, width=128, height=128, antialiasing=1.0)


parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--seed', type=int, default=2018)
args = parser.parse_args()

random.seed(args.seed)

os.makedirs('assets', exist_ok=True)

print("Rendering scenes...")
for color in colors:
    for object_type in object_types:
        for i in range(args.n_samples):
            filename = 'assets/%s-%s-%d' % (color, object_type, i)
            if os.path.exists(filename):
                print("%s exists, skipping" % filename)
                continue
            location = [random.uniform(-3, 3), random.uniform(-3, 3)]
            rotation = random.uniform(0, 360)
            render_scene(filename, object_type, color, location, rotation)

print("Finished")
