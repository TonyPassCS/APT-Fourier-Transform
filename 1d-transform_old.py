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


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.view_init(10, 0)


#Make 1 Dimension Line of length line_length, direction line-direction
def plot_line(line_direction, line_length):
    line = line_direction * line_length

    origin = np.array([0, 0, 0])
    xline = np.array([0, line[0]])
    yline = np.array([0, line[1]])
    zline = np.array([0, line[2]])
    ax.plot3D(xline, yline, zline, 'red')
    
    return line


line_direction = np.array([1, 1, 1])
line_length = 10

line = plot_line(line_direction, line_length)


#Make perfect cubic lattice of points of dimensions = lat_x, lat_y, lat_z, returns an array with these points and plots them
def make_perfect_lattice(lat_x, lat_y, lat_z):
    x_arr = np.arange(0, lat_x, 1)
    y_arr = np.arange(0, lat_y, 1)
    z_arr = np.arange(0, lat_z, 1)
    perf_lat_arr = []
    for x in x_arr:
        for y in y_arr:
            for z in z_arr:
                perf_lat_arr.append([x, y, z])
                ax.plot3D(x, y, z, '.')
                
    return perf_lat_arr


lat_x = 10
lat_y = 10
lat_z = 10
perf_lat_arr = np.array(make_perfect_lattice(lat_x, lat_y, lat_z))

#Generate random points for testing
n_points = 1000
rand_range = 10
rand_arr = []

for point_ in range(n_points):
    rand_point = random.randint(0, rand_range, size = 3)
    rand_arr.append(rand_point)

#put this in place of perf_latt_arr in order to test random control
rand_arr = np.array(rand_arr)

#Extract the bin coordinates for nearest bin calculations
def extract_bin_coordinates(n_interval, line_length, line_direction):
    interval_arr = np.array([n_interval,n_interval,n_interval])
    line_coord = []
    coord = np.array([0, 0, 0])
    counter_arr = []
    for i in np.arange(0, line_length, n_interval):
        line_coord.append(coord)
        coord = coord + line_direction*interval_arr
        counter_arr.append([0])
    line_coord = np.array(line_coord)
    counter_arr = np.array(counter_arr)
    
    return line_coord, counter_arr


#Set the number of bins
n_bins = 100
bin_arr = np.array(range(0, n_bins))
n_interval = line_length/n_bins
#gather coordinate for each bin
bin_coords, counter_arr = extract_bin_coordinates(n_interval, line_length, line_direction)

def assign_to_bins(point_arr, bin_coords, counter_arr):
    for point in point_arr:
        #dist is arbitrary, must simply be greater than the max distance possible in the model
        dist = 10000
        bin_no = 0
        bin_max = 0
        for bin_ in bin_coords:
            dist_2 = np.linalg.norm(bin_-point)
            if dist_2 < dist:
                dist = dist_2
                bin_max = bin_no
            bin_no += 1
        counter_arr[bin_max] = counter_arr[bin_max] + 1
    return counter_arr

bin_assign = assign_to_bins(perf_lat_arr, bin_coords, counter_arr)
print(bin_assign)


plt.figure(2)
plt.plot(bin_arr, bin_assign)

counter_arr_fft = np.fft.fft(counter_arr)
print(counter_arr_fft)
plt.figure(3)
plt.scatter(counter_arr_fft, bin_assign)