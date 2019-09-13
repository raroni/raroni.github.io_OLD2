import numpy as np
from numpy import pi, cos, sin, sqrt, arange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gen_spherical():
  # Create a sphere
  r = 1
  N = 800
  golden_ratio = (1 + 5**0.5) / 2.0
  indices = arange(0, N, dtype=float)
  # us = np.random.uniform(0, 1, (500, 500))
  us = [
    (indices + 0.5) / N, # push by 0.5 to "center" sample point
    np.mod(indices * golden_ratio, 1.0)
  ]

  pi = np.pi
  cos = np.cos
  sin = np.sin
  phi = np.arccos(1 - 2 * us[0])
  theta = 2 * np.pi * us[1]
  x = r*sin(phi)*cos(theta)
  y = r*sin(phi)*sin(theta)
  z = r*cos(phi)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, s=40)

  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.set_zlim([-1,1])
  ax.set_aspect("equal")
  plt.tight_layout()
  plt.savefig("spherical.svg", bbox_inches = 'tight')

gen_spherical()
