import glob
import os
import sys
import time

from end_to_end import nvidia_model
from keras.optimizers import Adam
from keras.models import load_model


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import numpy as np
import cv2


IM_WIDTH = 640
IM_HEIGHT = 480

lr = 1e-3
epochs = 10
width = 200
height = 66



model_name = 'Models/carla-{}-{}-{}-epochs.h5'.format(lr, 'nvidiaETE', epochs)

model = nvidia_model(width, height, 3)
model.compile(optimizer = Adam(lr=lr), loss='mse')
model = load_model(model_name)
model._make_predict_function()


def process_img(image, vehicle):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    screen = i2[:, :, :3]

    screen = cv2.resize(screen, (width,height))

    cv2.imshow("", screen)
    cv2.waitKey(1)


    data = model.predict(screen.reshape(-1,200,66,3), batch_size=1)

    print('throw: ' + str(data[0][0]) + '\t steer: ' + str(data[0][1]))
    #ewprint()

    control = carla.VehicleControl(throttle = float(0.3), steer = float(data[0][1]))
    vehicle.apply_control(control)
    '''
    our_vehicle_controll = vehicle.get_control()
    our_vehicle_controll.throttle = data[0][0]
    our_vehicle_controll.steer = data[0][1]
    '''

actor_list = []
vehicles_list = []

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=10.0,
        sun_altitude_angle=70.0)
    world.set_weather(weather)

    blueprint_library = world.get_blueprint_library()
    blueprints = world.get_blueprint_library().filter('vehicle')


    our_vehicle_bp = blueprint_library.filter('model3')[0]


    our_vehicle_spawn_point = world.get_map().get_spawn_points()[5]

    our_vehicle = world.spawn_actor(our_vehicle_bp, our_vehicle_spawn_point)
    #our_vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
    print('Auto dodat')
    actor_list.append(our_vehicle)




    time.sleep(5)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')


    print('Kamera dodata')
    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=our_vehicle)
    actor_list.append(sensor)

    sensor.listen(lambda data: process_img(data, our_vehicle))

    print('Krecemo')
    while True:
        world.tick()



        #print('tick: ' + str(t))


finally:
    sensor.stop()

    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    for actor in actor_list:
        actor.destroy()
    print('done.')
