import glob
import os
import sys
import time
import keyboard

from functional_conditional_end_to_end_keras_model import nvidia_model
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


IM_WIDTH = 200
IM_HEIGHT = 150

lr = 1e-4
epochs = 4
width = 200
height = 66

h_control = 0

#carla-0.0001-nvidiaETE-4-epochs-batches-aug-2nd.h5
#carla-0.001-nvidiaETE-12-epochs-batches.h5
model_name = 'Models/carla-0.001-nvidiaETE-10-epochs-batches.h5'.format(lr, 'nvidiaETE', epochs)

model = nvidia_model(width, height, 3, 4)
#model.compile(optimizer = Adam(lr=lr), loss='mse')
model = load_model(model_name)
# ovo u sustini treba da sredi neki problem koji bi se desio kada prvi put pozovem predict
# kao inicijalizacija predicta
model._make_predict_function()


def process_img(image, vehicle, h_control):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    screen = i2[-66:, :, :3]

    cv2.imshow("", screen)
    cv2.waitKey(1)


    one_hot = np.zeros((4))
    one_hot[h_control] = 1

    #print(one_hot.reshape(-1,4))

    data = model.predict([screen.reshape(-1,66,200,3), one_hot.astype(int).reshape(-1,4)], batch_size = 1)

    #print('steer: ' + str(data))
    #ewprint()

    control = carla.VehicleControl(throttle = float(0.5), steer = float(data))
    vehicle.apply_control(control)
    '''
    our_vehicle_controll = vehicle.get_control()
    our_vehicle_controll.throttle = data[0][0]
    our_vehicle_controll.steer = data[0][1]
    '''

actor_list = []
vehicles_list = []

town_name = input('Name of town: ')

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    if town_name != 'q':
        world = client.load_world(town_name)




    world = client.get_world()
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=10.0,
        sun_altitude_angle=70.0,
        precipitation_deposits=10)
    world.set_weather(weather)

    ''' cloudiness=80.0,
        precipitation=80.0,
        sun_altitude_angle=30.0,
        precipitation_deposits=80)  '''

    blueprint_library = world.get_blueprint_library()
    blueprints = world.get_blueprint_library().filter('vehicle')


    our_vehicle_bp = blueprint_library.filter('model3')[0]


    our_vehicle_spawn_point = world.get_map().get_spawn_points()[80]

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

    sensor.listen(lambda data: process_img(data, our_vehicle, h_control))

    print('Krecemo')
    while True:
        world.tick()

        if keyboard.is_pressed('left'):
            h_control = 1
            print('At the next intersection we go left.')
        elif keyboard.is_pressed('up'):
            h_control = 2
            print('At the next intersection we go straight.')
        elif keyboard.is_pressed('right'):
            h_control = 3
            print('At the next intersection we go right.')
        elif keyboard.is_pressed('down'):
            h_control = 0
            print('At the next intersection we go wherever we want.')



        #print('tick: ' + str(t))


finally:
    sensor.stop()

    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    for actor in actor_list:
        actor.destroy()
    print('done.')
