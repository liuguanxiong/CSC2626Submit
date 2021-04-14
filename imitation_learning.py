import torch
import argparse
import numpy as np
from PIL import Image
from carla.agent import Agent
from carla.carla_server_pb2 import Control
from model_check import ConditionalNet
import cv2
import subprocess

REACH_GOAL = 0.0
GO_STRAIGHT = 5.0
TURN_RIGHT = 4.0
TURN_LEFT = 3.0
LANE_FOLLOW = 2.0

class ImitationAgent(Agent):
    def __init__(self, city_name, avoid_stopping, image_cut=[115, 510]):
        Agent.__init__(self)
        self.image_cut = image_cut
        self.model = ConditionalNet()
        checkpoint = torch.load("save_models/training_best.pth")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.cuda()
        self.model.eval()
        self._avoid_stopping = avoid_stopping
        self.input_image_size = (200, 88)
    
    def run_step(self, measurements, sensor_data, directions, target):
        control = self.compute_action(sensor_data['CameraRGB'].data,
            measurements.player_measurements.forward_speed, directions)
        print(directions)
        return control

    def compute_action(self, rgb_image, speed, direction=None):
        rgb_image = rgb_image
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("temp.jpg", rgb_image)
        image_input = np.array(Image.fromarray(rgb_image).resize(size=self.input_image_size))
        image_input = np.multiply(image_input, 1.0 / 255.0)
        # while 1:
        #     traffic_light = subprocess.run("python ./traffic/detect.py --source {} --weights ./traffic/weights/best_model_12.pt --device cpu".format("./temp.jpg"),
        #                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     if not 'stop' in str(traffic_light.stdout):
        #         break
        steer, gas, brake = self.control_function(image_input, speed, direction)

        if brake < 0.1:
            brake = 0.0

        if gas > brake:
            brake = 0.0

        if speed > 10.0 and brake == 0.0:
            gas = 0.0

        control = Control()
        control.steer = steer
        control.throttle = gas
        control.brake = brake
        control.hand_brake = 0
        control.reverse = 0
        return control

    def control_function(self, image_input, speed, direction):
        image_input = torch.tensor(image_input.astype(np.float32))
        speed = speed / 90.0
        speed = torch.tensor(speed)
        image_input = image_input.unsqueeze(0).permute(0, 3, 1, 2)

        speed = speed.reshape(1, 1)
        image_input = image_input.cuda()
        speed = speed.cuda()
        output, pred_speed = self.model(image_input, speed)
        output = output.reshape((4,3)).cpu().detach().numpy()
        pred_speed = pred_speed.cpu().detach().numpy()
        if direction == LANE_FOLLOW or direction == REACH_GOAL:
            ret = output[0]
        elif direction == TURN_LEFT:
            ret = output[1]
        elif direction == TURN_RIGHT:
            ret = output[2]
        else:
            ret = output[3]

        steer = ret[0]
        gas = ret[1]
        brake = ret[2]
        if self._avoid_stopping:

            speed = speed.cpu().detach().numpy()[0][0]
            real_speed = speed
            real_predicted = pred_speed
            if real_speed < 2.0 / 90.0 and real_predicted > 3.0 / 90.0:
                # If (Car Stooped) and
                #  ( It should not have stopped, use the speed prediction branch for that)

                gas = 1 * (5.6 / 90.0 - speed) + gas

                brake = 0.0

        return steer, gas, abs(brake)