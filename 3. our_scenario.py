import glob
import os
import sys
import time

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



def process_img(image, training_data, vehicle):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    screen = i2[:, :, :3]
    cv2.imshow("", screen)
    cv2.waitKey(1)

    our_vehicle_controll = vehicle.get_control()
    throttle = our_vehicle_controll.throttle
    steer = our_vehicle_controll.steer

    control = [throttle, steer]
    training_data.append([screen,control])

    if len(training_data) % 25 == 0:
        print(len(training_data))


file_index = 0
file_name = input('Name of train file:')
file_name_index  = str(file_name) + '-' + str(file_index) + '.npy'


if os.path.isfile(file_name_index):
    print('File exists, we will overwritte a file!')
    training_data = []
else:
    print('File does not exist, starting fresh!')
    training_data = []


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


    our_vehicle_spawn_point = random.choice(world.get_map().get_spawn_points())

    our_vehicle = world.spawn_actor(our_vehicle_bp, our_vehicle_spawn_point)
    our_vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
    print('Auto dodat')
    actor_list.append(our_vehicle)



    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)



    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    print('prosao')
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= 120:            #args.number_of_vehicles
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

    for response in client.apply_batch_sync(batch):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    print('Dodati ostali automobili')

    time.sleep(5)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('sensor_tick', '0.2')

    print('Kamera dodata')
    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=our_vehicle)
    actor_list.append(sensor)

    sensor.listen(lambda data: process_img(data, training_data, our_vehicle))

    print('Krecemo')
    while True:
        world.tick()


        if len(training_data) > 2000:
            print('saving data')
            np.save(file_name_index,training_data)
            print(str(file_name_index) + 'is saved!')


            file_index += 1
            file_name_index = str(file_name) + '-' + str(file_index) + '.npy'

            training_data = []

        #print('tick: ' + str(t))


finally:
    print('usli u finalnu')
    sensor.stop()

    print('saving data')
    np.save(file_name_index,training_data)
    print(str(file_name_index) + 'is saved!')

    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    for actor in actor_list:
        actor.destroy()
    print('done.')
