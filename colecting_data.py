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
import pygame
from pygame.locals import *

IM_WIDTH = 200
IM_HEIGHT = 150


# function teleport vehicle to colect data with noise
def add_noise_teleport(vehicle):

    transform = vehicle.get_transform()

    if np.absolute(vehicle.get_control().steer) < 0.02 and np.absolute(transform.rotation.pitch) < 1:

        print(transform.location)

        angle = transform.rotation.yaw * np.pi / 180.
        print(angle)
        distance = (np.random.random() * 2 - 1) * 1.9         # distance -1.5 m < d < 1.5 m
        #distance = 1.5

        location = carla.Location(transform.location.x + np.sin(angle) * distance, transform.location.y + np.cos(angle) * distance, transform.location.z)
        rotation = carla.Rotation(transform.rotation.pitch, transform.rotation.yaw + (np.random.random() * 2 - 1) * 5, transform.rotation.roll)

        #print(np.absolute(transform.rotation.pitch))
        print(location)
        vehicle.set_transform(carla.Transform(location, rotation))


# function procces data for storing
def process_img(image, training_data, vehicle, display):
    raw_image = np.array(image.raw_data)
    image_bgra = raw_image.reshape((IM_HEIGHT, IM_WIDTH, 4))
    image_bgr = image_bgra[:, :, :3]
    #cv2.imshow("", image_bgr)
    #cv2.waitKey(20)
    r_image_bgr = np.transpose(image_bgr, axes = (1,0,2))

    rsurface = pygame.surfarray.make_surface(r_image_bgr)
    display.blit(rsurface, (0, 0))


    our_vehicle_controll = vehicle.get_control()
    throttle = our_vehicle_controll.throttle
    steer = our_vehicle_controll.steer


    text_surface_throttle = myfont.render('throtlee: '+str(throttle), False, (0, 0, 0))
    text_surface_steer = myfont.render('steer: '+str(steer), False, (0, 0, 0))

    display.blit(text_surface_throttle, (20, 20))
    display.blit(text_surface_steer, (20, 50))
    pygame.display.flip()

    #print(np.absolute(vehicle.get_transform().rotation.yaw))

    control = [throttle, steer]
    training_data.append([image_bgr,control])

    if len(training_data) % 25 == 0:
        print(len(training_data))


# definig baches name
file_index = 0
name_of_colecting_sesion = input('Name of colecting session:')
file_name_index  = 'Data/' + str(name_of_colecting_sesion) + '_' + str(file_index) + '.npy'

pygame.init()
pygame.font.init()

myfont = pygame.font.SysFont('Comic Sans MS', 15)
display = pygame.display.set_mode((IM_WIDTH,IM_HEIGHT))


if os.path.isfile(file_name_index):
    print('File exists, we will overwritte a file!')
    training_data = []
else:
    print('File does not exist, starting fresh!')
    training_data = []


actor_list = []
vehicles_list = []

try:
    # defining Carla world
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=10.0,
        sun_altitude_angle=70.0)
    world.set_weather(weather)


    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.freeze(True)


    # defining out vehicle
    blueprint_library = world.get_blueprint_library()
    blueprints = world.get_blueprint_library().filter('vehicle')


    our_vehicle_bp = blueprint_library.filter('model3')[0]


    our_vehicle_spawn_point = random.choice(world.get_map().get_spawn_points())

    our_vehicle = world.spawn_actor(our_vehicle_bp, our_vehicle_spawn_point)
    our_vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
    print('Car is append!')
    actor_list.append(our_vehicle)



    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)


    # we can add other vehicles here


    time.sleep(2)


    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # creating camera sensor

    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    #blueprint.set_attribute('sensor_tick', '0.2')

    print('Camera is apend!')
    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=our_vehicle)
    actor_list.append(sensor)

    sensor.listen(lambda data: process_img(data, training_data, our_vehicle, display))

    print('Car driving, loop start!')


    #   MAIN WHILE LOOP   #
    done = False
    while not done:

        #exiting of pygame
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    print('Ending...')
                    break

        # world tick
        t = world.tick()


        if t % 50 == 0:
            add_noise_teleport(our_vehicle)
            print('tick: ' + str(t))

        # saving baches of data
        if len(training_data) == 256:
            print('saving data...')

            temp = training_data
            training_data = []

            np.save(file_name_index, temp)
            print(str(file_name_index) + 'is saved!')


            file_index += 1
            file_name_index = 'Data/' + str(name_of_colecting_sesion) + '_' + str(file_index) + '.npy'



finally:
    sensor.stop()
    pygame.quit()


    print('Saving data...')
    np.save(file_name_index,training_data)
    print(str(file_name_index) + 'is saved!')

    print('Destroying actors...')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    for actor in actor_list:
        actor.destroy()

    print('Done...')
