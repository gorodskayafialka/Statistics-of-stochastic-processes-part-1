import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.stats import norm
import numpy.random as rnd
import csv


def generater_OU(n, t, x, theta):
  Ex = theta[0]/theta[1] + (x - theta[0]/theta[1]) * np.exp(-theta[1]*t)
  Vx = theta[2] ** 2 * np.sqrt((1-np.exp(-2*theta[1]*t))/(2*theta[1]))
  return np.random.normal (Ex, np.sqrt(Vx), n)

def generater_BS(n, dt, x, theta):
  lmean = np.log(x) + (theta[0]-0.5*theta[1]**2)*dt
  lsd = np.sqrt(dt)*theta[1]
  return np.random.lognormal(lmean, lsd, n)

def generater_CIR(n, dt, t, x, theta):
  c = 2*theta[1] /((1-np.exp(-theta[1]*t))* theta[2] ** 2)
  ncp = 2 * c * x * np.exp(-theta[2]*dt)
  df = 4 * theta[0]/theta[2]**2
  return np.random.noncentral_chisquare(df, ncp, n) /(2 * c)

with open("C:\\Users\\Ксения Лучкова\\OneDrive\\Документы\\Сириус\\Семестр 1\\Статистика случайных процессов\\3_3_a.csv", 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=' ', quotechar=',', quoting=csv.QUOTE_MINIMAL)
  for t in range(0, 100):
    writer.writerow(generater_OU(1, t, 10, [2, 0.2, 0.15]))


with open("C:\\Users\\Ксения Лучкова\\OneDrive\\Документы\\Сириус\\Семестр 1\\Статистика случайных процессов\\3_3_b.csv", 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=' ', quotechar=',', quoting=csv.QUOTE_MINIMAL)
  for t in range(0, 100):
    writer.writerow(generater_BS(1, 0.1, 10, [2, 0.2, 0.15]))

 
with open("C:\\Users\\Ксения Лучкова\\OneDrive\\Документы\\Сириус\\Семестр 1\\Статистика случайных процессов\\3_3_c.csv", 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=' ', quotechar=',', quoting=csv.QUOTE_MINIMAL)
  for t in range(0, 100):
    writer.writerow(generater_CIR(1, 0.1, t, 10, [2, 0.2, 0.15]))