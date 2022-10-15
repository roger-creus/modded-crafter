import numpy as np
import cv2
import os
import random

size = 64, 64
duration = 500
fps = 30
observations_path = 'src/trajectories/observations/'

obs = np.load(observations_path + "trajectory_observations_" + str(0) + ".npy")
step = 0
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for _ in range(fps * duration):
    img = obs[step].squeeze()
    img = (img*255).astype(np.uint8)
    out.write(img)
    step += 1

    if step >= len(obs):
        print("finished")
        break
out.release()
