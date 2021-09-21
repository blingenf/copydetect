import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_training_data_binary(num):
  if num == 1:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, -1]

  elif num == 2:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [0, i-5, 1]
      data[i+5] = [0, i+1, -1]

  elif num == 3:
    data = np.zeros((10,3))
    data[0] = [3, 2, 1]
    data[1] = [6, 2, 1]
    data[2] = [0, 6, 1]
    data[3] = [4, 4, 1]
    data[4] = [5, 4, 1]
    data[5] = [-1, -2, -1]
    data[6] = [-2, -4, -1]
    data[7] = [-3, -3, -1]
    data[8] = [-4, -2, -1]
    data[9] = [-4, -4, -1]
  elif num == 4:
    data = np.zeros((10,3))
    data[0] = [-1, 1, 1]
    data[1] = [-2, 2, 1]
    data[2] = [-3, 5, 1]
    data[3] = [-3, -1, 1]
    data[4] = [-2, 1, 1]
    data[5] = [3, -6, -1]
    data[6] = [0, -2, -1]
    data[7] = [-1, -7, -1]
    data[8] = [1, -10, -1]
    data[9] = [0, -8, -1]

  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return data

def generate_training_data_multi(num):
  if num == 1:
    data = np.zeros((20,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, 2]
      data[i+10] = [0, i-5, 3]
      data[i+15] = [0, i+1, 4]
    Y = 4

  elif num == 2:
    data = np.zeros((15,3))
    data[0] = [-5, -5, 1]
    data[1] = [-3, -2, 1]
    data[2] = [-5, -3, 1]
    data[3] = [-5, -4, 1]
    data[4] = [-2, -9, 1]
    data[5] = [0, 6, 2]
    data[6] = [-1, 3, 2]
    data[7] = [-2, 1, 2]
    data[8] = [1, 7, 2]
    data[9] = [1, 5, 2]
    data[10] = [6, 3, 3]
    data[11] = [9, 2, 3]
    data[12] = [10, 4, 3]
    data[13] = [8, 1, 3]
    data[14] = [9, 0, 3]
    Y = 3

  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return [data, Y]

def plot_training_data_binary(data, boundary=None):
  for item in data:
    if item[-1] == 1:
      plt.plot(item[0], item[1], 'b+')
    else:
      plt.plot(item[0], item[1], 'ro')
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])

  if boundary:
    w = boundary[0]
    b = boundary[1]
    s = boundary[2]

    if (w[0] == 0):
        x2_min = -(-m*w[0]+b)/(w[1])
        x2_max = -( m*w[0]+b)/(w[1])

        diff = -(s[0][0]*w[0]+b)/(w[1]) - s[0][1]
        plt.plot((-m, m), (x2_min+diff, x2_max+diff), color='black',ls='dashed')
        plt.plot((-m, m), (x2_min-diff, x2_max-diff), color='black',ls='dashed')
        plt.plot((-m, m), (x2_min, x2_max), color='black')
    else:
        x1_min = -(-m*w[1]+b)/(w[0])
        x1_max = -( m*w[1]+b)/(w[0])

        diff = -(s[0][1]*w[1]+b)/(w[0]) - s[0][0]
        plt.plot((x1_min+diff, x1_max+diff), (-m, m), color='black',ls='dashed')
        plt.plot((x1_min-diff, x1_max-diff), (-m, m), color='black',ls='dashed')
        plt.plot((x1_min, x1_max), (-m, m), color='black')

    for sv in s:
        plt.scatter(sv[0], sv[1], s=200, facecolors='none', edgecolors='black')
  plt.show()

def plot_training_data_multi(data, boundaries=None, plot_margin=False):
  colors = ['b', 'r', 'g', 'm']
  shapes = ['+', 'o', '*', '.']

  for item in data:
    plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])

  if boundaries:
    for i in range(len(boundaries[0])):
      w = boundaries[0][i]
      b = boundaries[1][i]
      s = boundaries[2][i]

      if (w[0] == 0):
          x2_min = -(-m*w[0]+b)/(w[1])
          x2_max = -( m*w[0]+b)/(w[1])

          if plot_margin:
              diff = -(s[0][0]*w[0]+b)/(w[1]) - s[0][1]
              plt.plot((-m, m), (x2_min+diff, x2_max+diff), color='black',ls='dashed')
              plt.plot((-m, m), (x2_min-diff, x2_max-diff), color='black',ls='dashed')
          plt.plot((-m, m), (x2_min, x2_max), color='black')
      else:
          x1_min = -(-m*w[1]+b)/(w[0])
          x1_max = -( m*w[1]+b)/(w[0])

          if plot_margin:
              diff = -(s[0][1]*w[1]+b)/(w[0]) - s[0][0]
              plt.plot((x1_min+diff, x1_max+diff), (-m, m), color='black',ls='dashed')
              plt.plot((x1_min-diff, x1_max-diff), (-m, m), color='black',ls='dashed')
          plt.plot((x1_min, x1_max), (-m, m), color='black')
      if plot_margin:
          for sv in s:
              plt.scatter(sv[0], sv[1], s=200, facecolors='none', edgecolors='black')

  plt.show()
