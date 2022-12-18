from logging import Logger
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import json
import math
import sys
import argparse
import shutil
import numpy as np
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from torch import nn
from transformer import Transformer
from dataset import euclidean_distance
import torch.nn.functional as f
from torch.distributions import Categorical

parameters = [['0', 'a', 2, 16, 480, 3, 30],  # 0
              ['1', 'a', 2, 20, 600, 3, 30],  # 1
              ['2', 'a', 2, 24, 720, 3, 30],  # 2
              ['4', 'a', 3, 24, 480, 3, 30],  # 3
              ['6', 'a', 3, 36, 720, 3, 30],  # 4
              ['9', 'a', 4, 32, 480, 3, 30],  # 5
              ['10', 'a', 4, 40, 600, 3, 30],  # 6
              ['11', 'a', 4, 48, 720, 3, 30],  # 7
              ['24', 'b', 2, 16, 480, 6, 45],  # 8
              ['25', 'b', 2, 20, 600, 6, 45],  # 9
              ['26', 'b', 2, 24, 720, 6, 45],  # 10
              ['28', 'b', 3, 24, 480, 6, 45],  # 11
              ['30', 'b', 3, 36, 720, 6, 45],  # 12
              ['33', 'b', 4, 32, 480, 6, 45],  # 13
              ['34', 'b', 4, 40, 600, 6, 45],  # 14
              ['35', 'b', 4, 48, 720, 6, 45]]  # 15




def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_instances', type=int, default=100)
    parser.add_argument('--index', type=int, default=9)
    parser.add_argument('--mask', type=str, default='off')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    return args


class User:
    def __init__(self):
        self.id = 0
        self.max_ride_time = 0
        self.pickup_coords = []
        self.dropoff_coords = []
        self.pickup_window = []
        self.dropoff_window = []
        self.duration = 0
        self.load = 0
        # Status of a user taking values in {0, 1, 2}
        # 0: waiting
        # 1: being served
        # 2: done
        self.status = 0
        # Flag of a user taking values in {0, 1, 2}
        # 0: waiting
        # 1: being served by the vehicle which performs an action at time step t
        # 2: done or unable to be served
        self.flag = 0
        self.served_by = 0
        self.ride_time = 0.0
        self.served_id = []
    



class Vehicle:
    def __init__(self):
        self.id = 0
        self.max_route_duration = 0
        self.max_capacity = 0
        self.route = []
        self.schedule = []
        self.ordinal = 1
        self.coords = []
        self.free_capacity = 0
        self.ride_time = {}
        self.free_time = 0.0
        self.duration = 0
        self.pred_route = [0]
        self.pred_schedule = [0]
        self.cost = 0.0


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def shift_window(time_window, time):
    return [max(time_window[0] - time, 0.0), max(time_window[1] - time, 0.0)]


def check_window(time_window, time):
    return time < time_window[0] or time > time_window[1]


def update_ride_time(vehicle, users, ride_time):
    for key in vehicle.ride_time:
        vehicle.ride_time[key] += ride_time
        users[key - 1].ride_time += ride_time



class darpenv():
    def __init__(self,
                 model,
                 bmodel,
                 file,
                 size:int,
                 num_users:int,
                 num_vehicles:int,
                 time_end:int,
                 max_step:int,
                 max_route_duration: Optional[int]=None,
                 max_vehicle_capacity: Optional[int]=None,
                 capacity: Optional[int]=None,
                 max_ride_time: Optional[int]=None,
                 seed: Optional[int]=None,
                 window: Optional[bool]=None
                 ):
        super(darpenv, self).__init__()
        
        print("initializing env")
        self.size = size
        self.max_step = max_step
        self.num_users = num_users
        self.num_vehicles = num_vehicles
        self.max_vehicle_capcity = max_vehicle_capacity
        self.capacity = capacity
        self.time_end = time_end
        self.seed = seed
        self.current_episode = 0
        self.window = window 
        self.current_step = 0
        
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_instances', type=int, default=100)
        parser.add_argument('--index', type=int, default=9)
        parser.add_argument('--mask', type=str, default='off')
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--d_k', type=int, default=64)
        parser.add_argument('--d_v', type=int, default=64)
        parser.add_argument('--d_ff', type=int, default=2048)
        parser.add_argument('--dropout', type=float, default=0.1)
    
        args = parser.parse_args()

        instance_type = parameters[args.index][1]
        num_vehicles = parameters[args.index][2]
        num_users = parameters[args.index][3]
        max_route_duration = parameters[args.index][4]
        max_vehicle_capacity = parameters[args.index][5]
        max_ride_time = parameters[args.index][6]
        print('Number of vehicles: {}.'.format(num_vehicles),
              'Number of users: {}.'.format(num_users),
              'Maximum route duration: {}.'.format(max_route_duration),
              'Maximum vehicle capacity: {}.'.format(max_vehicle_capacity),
              'Maximum ride time: {}.'.format(max_ride_time))

        
        self.model = model
        self.bmodel = bmodel
        self.file=file
        self.nodes_to_users = {}
        for i in range(1, 2 * (self.num_users + 1) - 1):
            if i <= self.num_users:
                self.nodes_to_users[i] = i
            else:
                self.nodes_to_users[i] = i - self.num_users
        self.num_instance = 0
        self.eval_obj_true = []
        self.eval_obj_pred = []
        self.eval_window = []
        self.eval_ride_time = []
        self.eval_not_same = []
        self.eval_not_done = []
        self.eval_rela_gap = []
        if max_route_duration:
            self.max_route_duration = max_route_duration
        else:
            self.max_route_duration = self.max_step
        if max_ride_time:
            self.max_ride_time = max_ride_time
        else:
            self.max_ride_time = self.max_step
        self.users = []
        self.vehicles = []
        pairs=[]
        path = './instance/b2-20-train.txt'
        with open(path, 'r') as file:
            for pair in file:
                pairs.append(pair)
        self.pairs = pairs

        





    def reset (self,instance_number,relax_window=False):

        print("populate env instance with", self.num_vehicles," Vehicle and" ,self.num_users, "Users objects" )
        pair = json.loads(self.pairs[instance_number])

        num_vehicles = pair['instance'][0][0]
        num_users = pair['instance'][0][1]
        max_route_duration = pair['instance'][0][2]
        max_vehicle_capacity = pair['instance'][0][3]
        max_ride_time = pair['instance'][0][4]
        objective = pair['objective']
        self.time_penalties =[]
        """init users"""
        users = []
        for i in range(1, num_users + 1):
            user = User()
            user.id = i
            user.max_ride_time = max_ride_time
            user.served_by = num_vehicles
            users.append(user)

        """init nodes"""
        for i in range(0, 2 * (num_users + 1)):
                node = pair['instance'][i + 1]
                if i == 0:
                    self.orig_depot_coords = [float(node[1]), float(node[2])]
                    continue
                if i == 2 * (num_users + 1) - 1:
                    self.dest_depot_coords = [float(node[1]), float(node[2])]
                    continue
                user = users[self.nodes_to_users[i] - 1]
                if i <= num_users:
                    # Pick-up nodes
                    user.pickup_coords = [float(node[1]), float(node[2])]
                    user.duration = node[3]
                    user.load = node[4]
                    user.pickup_window = [float(node[5]), float(node[6])]
                else:
                    # Drop-off nodes
                    user.dropoff_coords = [float(node[1]), float(node[2])]
                    user.dropoff_window = [float(node[5]), float(node[6])]


        for user in users:
                travel_time = euclidean_distance(user.pickup_coords, user.dropoff_coords)
                if user.id <= num_users / 2:
                    # Drop-off requests
                    user.pickup_window[0] = \
                        round(max(0.0, user.dropoff_window[0] - max_ride_time - user.duration), 3)
                    user.pickup_window[1] = \
                        round(min(user.dropoff_window[1] - travel_time - user.duration, max_route_duration), 3)
                else:
                    # Pick-up requests
                    user.dropoff_window[0] = \
                        round(max(0.0, user.pickup_window[0] + user.duration + travel_time), 3)
                    user.dropoff_window[1] = \
                        round(min(user.pickup_window[1] + user.duration + max_ride_time, max_route_duration), 3)

        """init vehicles"""
        vehicles = []
        for n in range(0, num_vehicles):
            vehicle = Vehicle()
            vehicle.id = n
            vehicle.max_capacity = max_vehicle_capacity
            vehicle.max_route_duration = max_route_duration
            vehicle.route = pair['routes'][n]
            vehicle.route.insert(0, 0)
            vehicle.route.append(2 * num_users + 1)
            vehicle.schedule = pair['schedule'][n]
            vehicle.coords = [0.0, 0.0]
            vehicle.free_capacity = max_vehicle_capacity
            vehicle.free_time = 0.0
            vehicles.append(vehicle)
        
        """init env"""
        self.break_window = []
        self.break_ride_time = []
        self.break_same = []
        self.break_done = []
        self.users = users
        self.vehicles = vehicles
        self.break_window = []
        self.break_ride_time = []
        self.break_same = []
        self.break_done = []
        self.wait_time =5
        self.current_step = 0

        return objective
        

        

    def get_vehicle_state(self,vehicle_id,time,device):
        for user in self.users:
            if user.status == 1 and user.served_by == vehicle_id:
                # 1: being served by the vehicle which performs an action at time step t
                user.flag = 1
            else:
                if user.status == 0:
                    if user.load <= self.vehicles[vehicle_id].free_capacity:
                        # 0: waiting
                        user.flag = 0
                    else:
                        # 2: unable to be served
                        user.flag = 2
                else:
                    # 2: done
                    user.flag = 2
        # User information.
        users_info = [list(map(np.float64,
                               [user.duration,
                                user.load,
                                user.status,
                                user.served_by,
                                user.ride_time,
                                shift_window(user.pickup_window, time),
                                shift_window(user.dropoff_window, time),
                                vehicle_id,
                                user.flag]
                               + [vehicle.duration + euclidean_distance(
                                   vehicle.coords, user.pickup_coords)
                                  if user.status == 0 else
                                  vehicle.duration + euclidean_distance(
                                      vehicle.coords, user.dropoff_coords)
                                  for vehicle in self.vehicles])) for user in self.users]
        # Mask information.
        # 0: waiting, 1: being served, 2: done
        mask_info = [0 if user.flag == 2 else 1 for user in self.users]
        state = [users_info, mask_info]
        state, _ = DataLoader([state, 0], batch_size=1)  # noqa
        mask = torch.Tensor(mask_info + [1, 1]).to(device)

        return state, mask

    

      
    def step(self,device,state,prediction,vehicle_id,indices):
        self.current_step +=1 
        time_pen =0 
        nb_time_pen = 0
        vehicle = self.vehicles[vehicle_id]
        if prediction == self.num_users + 1:
                        vehicle.free_time += self.wait_time
                        update_ride_time(vehicle, self.users, self.wait_time)
        else:
            if vehicle.pred_route[-1] != 0 and vehicle.pred_route[-1] != 25:
                            user = self.users[vehicle.pred_route[-1] - 1]
                            if user.id not in vehicle.ride_time.keys():
                                if check_window(user.pickup_window, vehicle.free_time) and user.id > self.num_users / 2:
                                    print('The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                                        user.id, vehicle.free_time, user.pickup_window))
                                    self.file.write('The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                                        user.id, vehicle.free_time, user.pickup_window))
                                    self.break_window.append(user.id)
                                    time_pen =+ vehicle.free_time - user.pickup_window[1]
                                    nb_time_pen=+1
                                vehicle.ride_time[user.id] = 0.0
                            else:
                                if user.ride_time - user.duration > self.max_ride_time + 1e-2:
                                    if user.id > self.num_users / 2 or vehicle.pred_route[-2] != vehicle.pred_route[-1]:
                                        print('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                                            user.id, user.ride_time - user.duration, self.max_ride_time))
                                        self.file.write('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                                            user.id, user.ride_time - user.duration, self.max_ride_time))
                                        self.break_ride_time.append(user.id)
                                        time_pen =+ user.ride_time - user.duration - self.max_ride_time
                                        nb_time_pen =+1
    
                                if check_window(user.dropoff_window, vehicle.free_time) and user.id <= self.num_users / 2:
                                    print('The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                                        user.id, vehicle.free_time, user.dropoff_window))
                                    self.file.write('The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                                        user.id, vehicle.free_time, user.dropoff_window))
                                    self.break_window.append(user.id)
                                    time_pen =+ vehicle.free_time - user.dropoff_window[1]
                                    nb_time_pen =+1
                                del vehicle.ride_time[user.id]

                            vehicle.duration = user.duration
                            user.ride_time = 0.0

            if prediction < self.num_users:
                user = self.users[prediction]
                if user.id not in vehicle.ride_time.keys():
                    travel_time = euclidean_distance(vehicle.coords, user.pickup_coords)
                    window_start = user.pickup_window[0]
                    vehicle.coords = user.pickup_coords
                    vehicle.free_capacity -= user.load
                    user.served_by = vehicle.id
                    user.served_id.append(vehicle.id)
                    user.status = 1
                else:
                    travel_time = euclidean_distance(vehicle.coords, user.dropoff_coords)
                    window_start = user.dropoff_window[0]
                    vehicle.coords = user.dropoff_coords
                    vehicle.free_capacity += user.load
                    user.served_by = self.num_vehicles
                    user.served_id.append(vehicle.id)
                    user.status = 2


                if vehicle.free_time + vehicle.duration + travel_time > window_start + 1e-2:
                    ride_time = vehicle.duration + travel_time
                    vehicle.free_time += ride_time
                else:
                    ride_time = window_start - vehicle.free_time
                    vehicle.free_time = window_start
                vehicle.cost += travel_time
                update_ride_time(vehicle, self.users, ride_time)
            else:
                vehicle.cost += euclidean_distance(vehicle.coords, self.dest_depot_coords)
                vehicle.coords = self.dest_depot_coords
                vehicle.free_time = int(self.max_route_duration)
                vehicle.duration = 0
            vehicle.pred_route.append(prediction.item() + 1)
            vehicle.pred_schedule.append(vehicle.free_time)
        
        print(vehicle_id)
        print (self.vehicles[vehicle_id].pred_route)



        return time_pen , nb_time_pen

        

    def finish (self):
        free_times = [vehicle.max_route_duration - vehicle.free_time  for vehicle in self.vehicles]
        free_times = np.array(free_times)
        free_times = free_times[free_times <= 0]

        

        if len(free_times) == self.num_vehicles:
            for user in self.users:
                # Check if the user is served by the same vehicle.
                if len(user.served_id) != 2 or user.served_id[0] != user.served_id[1]:
                    self.break_same.append(user.id)
                    print('* User {} is served by {}.'.format(user.id, user.served_id))
                    self.file.write('* User {} is served by {}.'.format(user.id, user.served_id))
                # Check if the request of the user is finished.
                if user.status != 2:
                    self.break_done.append(user.id)
                    print('* The request of User {} is unfinished.'.format(user.id))
                    self.file.write('* The request of User {} is unfinished.'.format(user.id))
            for vehicle in self.vehicles:
                print('> Vehicle {}'.format(vehicle.id))
                self.file.write('> Vehicle {}'.format(vehicle.id))
                for index, node in enumerate(vehicle.route):
                    if 0 < node < 2 * self.num_users + 1:
                        vehicle.route[index] = self.nodes_to_users[node]
                ground_truth = zip(vehicle.route[1:-1], vehicle.schedule[1:-1])
                prediction = zip(vehicle.pred_route[1:-1], vehicle.pred_schedule[1:-1])
                print('Ground truth: {}'.format([term[0] for term in ground_truth]))
                self.file.write('Ground truth: ' )
                print('Prediction:', [term[0] for term in prediction])




        return len(free_times)== self.num_vehicles

    def get_cost(self): 
        return sum(vehicle.cost for vehicle in self.vehicles)    

def main():
    args = parse_arguments()
    instance_type = parameters[args.index][1]
    num_vehicles = parameters[args.index][2]
    num_users = parameters[args.index][3]
    max_route_duration = parameters[args.index][4]
    max_vehicle_capacity = parameters[args.index][5]
    max_ride_time = parameters[args.index][6]
    print('Number of vehicles: {}.'.format(num_vehicles),
          'Number of users: {}.'.format(num_users),
          'Maximum route duration: {}.'.format(max_route_duration),
          'Maximum vehicle capacity: {}.'.format(max_vehicle_capacity),
          'Maximum ride time: {}.'.format(max_ride_time))
 
    env = darpenv(size =10 , num_users=24, num_vehicles=3, time_end=1400, max_step=100)



if __name__ == "__main__":
    main()