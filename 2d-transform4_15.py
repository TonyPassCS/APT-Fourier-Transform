import numpy as np
from numpy import random
from scipy import interpolate, optimize, special
import math
#import MieFunctions_2 as mie
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as cols
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import struct
import re
from sympy import Plane, Point, Point3D
from PIL import Image

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.view_init(10, 0)

#Make perfect cubic lattice of points of dimensions = lat_x, lat_y, lat_z, returns an array with these points and plots them
def make_perfect_lattice(lat_x, lat_y, lat_z):
    x_arr = np.arange(-lat_x, lat_x+0.25, 0.25)
    y_arr = np.arange(-lat_y, lat_y+0.25, 0.25)
    z_arr = np.arange(-lat_z, lat_z+0.25, 0.25)
    perf_lat_arr = []
    for x in x_arr:
        for y in y_arr:
            for z in z_arr:
                perf_lat_arr.append([x, y, z])
                ax.plot3D(x, y, z, '.')
                
    return perf_lat_arr


lat_x = 1
lat_y = 1
lat_z = 1
perf_lat_arr = np.array(make_perfect_lattice(lat_x, lat_y, lat_z))


#Generate random points for testing
def generate_random_data(n_points):
    rand_arr = []

    for point_ in range(n_points):
        rand_point = random.uniform(-1, 1, size = 3)
        rand_arr.append(rand_point)

    #put this in place of perf_latt_arr in order to test random control
    rand_arr = np.array(rand_arr)
    return rand_arr

n_points = 1000
rand_arr = generate_random_data(n_points)

#Projection onto 2D plane

origin = [0, 0, 0]
normal = [1, 1, 1]
proj_arr = []

for point in perf_lat_arr:
    emp_arr = []
    p = Point3D(point)
    plane = Plane(Point3D(origin), normal_vector = normal)
    proj_point = plane.projection(p)
    emp_arr.append(float(proj_point[0]))
    emp_arr.append(float(proj_point[1]))
    emp_arr.append(float(proj_point[2]))
    proj_arr.append(emp_arr)
proj_arr = np.array(proj_arr)

plt.figure(2, figsize = (15,15))
hist = plt.hist2d(proj_arr[:,0], proj_arr[:,1], bins = 200)
hist_arr = np.asarray(hist[0])




fft = np.fft.fftshift(np.fft.fft2(hist_arr))
plt.figure(figsize=(9, 9))

abs_hist = []
for arr in fft:
    abs_arr = []
    for val in arr:
        val = abs(val)
        abs_arr.append(val)
    abs_hist.append(abs_arr)

plt.imshow(abs_hist)
ax.set_aspect('auto')
plt.axis('off')
plt.show()


