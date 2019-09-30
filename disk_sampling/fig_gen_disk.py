import numpy as np
from numpy import pi, cos, sin, sqrt, arange
import matplotlib.pyplot as plt

# For determinism.
plt.rcParams['svg.hashsalt'] = 42
np.random.seed(0)

dot_size = 20
N_sqrt = 12
N = 12*12
golden_ratio = (1 + 5**0.5) / 2.0

def xs(rs, thetas):
  return rs*cos(thetas)

def ys(rs, thetas):
  return rs*sin(thetas)

def draw_subplot(idx, xs, ys, ylim, xlim, viz):
  plt.subplot(1, 2, idx, aspect = 'equal')
  plt.ylim(ylim)
  plt.xlim(xlim)
  plt.scatter(xs, ys, s = dot_size)

  if viz:
    theta = np.linspace(-np.pi, np.pi, 100)
    plt.plot(np.sin(theta), np.cos(theta), color = 'red')
    plt.scatter([0], [0], color = 'red', s = dot_size)

# Create two plots side by side.
def gen2(
  xs1, ys1, circle_viz1, ylim1, xlim1,
  xs2, ys2, circle_viz2, ylim2, xlim2,
  filename
):
  print("Creating " + filename)
  draw_subplot(1, xs1, ys1, ylim1, xlim1, circle_viz1)
  draw_subplot(2, xs2, ys2, ylim2, xlim2, circle_viz2)
  plt.tight_layout()
  plt.savefig(filename, bbox_inches = 'tight')
  plt.clf()

def gen2_cart_polar(us, filename):
  rs = np.sqrt(us[0])
  thetas = 2 * pi * us[1]
  gen2(
    us[0], us[1], False, (0, 1), (0, 1),
    xs(rs, thetas), ys(rs, thetas), True, (-1, 1), (-1, 1),
    filename
  )

def gen_naive():
  n = 400
  rs_uniform = np.random.uniform(0, 1, n)
  thetas_uniform = np.random.uniform(-np.pi, np.pi, n)
  rs_sqrooted = np.sqrt(rs_uniform)
  gen2(
    xs(rs_uniform, thetas_uniform), ys(rs_uniform, thetas_uniform), True, (-1, 1), (-1, 1),
    xs(rs_sqrooted, thetas_uniform), ys(rs_sqrooted, thetas_uniform), True, (-1, 1), (-1, 1),
    'naive.svg'
  )

def gen_uniform():
  us = np.random.uniform(0, 1, (N, N))
  gen2_cart_polar(us, 'uniform.svg')

def gen_regular():
  indices = arange(0, N, dtype=float)
  us = [
    np.floor(indices / N_sqrt) / N_sqrt + 0.5/N_sqrt,
    np.mod(indices, N_sqrt) / N_sqrt + 0.5/N_sqrt
  ]
  gen2_cart_polar(us, 'regular.svg')

def gen_irrational1():
  indices = arange(0, N, dtype=float)
  us = [
    np.floor(indices / N_sqrt) / N_sqrt + 0.5/N_sqrt,
    np.mod(indices * np.pi, 1.0)
  ]
  gen2_cart_polar(us, 'irrational1.svg')

def gen_golden_ratio1():
  indices = arange(0, N, dtype=float)
  us = [
    np.floor(indices / N_sqrt) / N_sqrt + 0.5/N_sqrt,
    np.mod(indices * golden_ratio, 1.0)
  ]
  gen2_cart_polar(us, 'golden_ratio1.svg')

def gen_final():
  indices = arange(0, N, dtype=float)
  us = [
    (indices + 0.5) / N, # push by 0.5 to "center" sample point
    np.mod(indices * golden_ratio, 1.0)
  ]
  gen2_cart_polar(us, 'final.svg')

gen_naive()
gen_uniform()
gen_regular()
gen_irrational1()
gen_golden_ratio1()
gen_final()
