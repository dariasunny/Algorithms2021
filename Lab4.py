import scipy
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import random

from __future__ import division

from scipy.spatial import distance_matrix

#Number 1
#our function
def f(x):
  return 1/(x**2 -3*x +2)
  
#our data
y_val = []
for k in range(0,1001):
  x = 3*k/1000
  delta = random.gauss(0,1)
  if f(x) < -100:
    y_val.append(-100+ delta)
  elif -100 <= f(x) and f(x) <= 100:
    y_val.append(f(x)+ delta)
  else:
    y_val.append(100 + delta)
   
def rational(x, a, b,c,d):
  return (a*x+b)/(x**2 + c*x +d)
  
def D(a,b,c,d,func, y):
  summa = 0
  for k in range(0, 1001):
    summa += (func(3*k/1000,a,b,c,d) - y[k])**2
  return summa
  
def D_copy(a,b,c,d, func, y):
  summa = 0
  for k in range(0, 1001):
    summa += (func(3*k/1000,a,b, c, d) - y[k])**2
  return [summa,summa, summa, summa]

#algorithm Nelder-Mead
res_rational = scipy.optimize.minimize(lambda a : D(a[0],a[1], a[2], a[3], rational, y_val), [0,0,0,0], method = 'Nelder-Mead', options={'fatol': 0.01})
res_rational

#algorithm LM
lm_func = scipy.optimize.root(lambda a : D_copy(a[0],a[1], a[2], a[3], rational, y_val), [0,0,0,0], method = 'lm', options={'fatol': 0.01})
lm_func

#algorithm Differential Evolution
de_func = scipy.optimize.differential_evolution(lambda a : D(a[0],a[1], a[2], a[3], rational, y_val), [(0,1),(0,1),(0,1),(0,1)], strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7)
de_func

#algorithm Annealing
sa_func = scipy.optimize.dual_annealing(lambda a : D(a[0],a[1], a[2], a[3], rational, y_val), [(0,1),(0,1),(0,1),(0,1)])
sa_func

#paramerets
a_plot_nm_rational, b_plot_nm_rational, c_plot_nm_rational , d_plot_nm_rational = res_rational.x[0], res_rational.x[1], res_rational.x[2], res_rational.x[3]
a_plot_lm_rational, b_plot_lm_rational, c_plot_lm_rational , d_plot_lm_rational = lm_func.x[0], lm_func.x[1], lm_func.x[2], lm_func.x[3]
a_plot_de_rational, b_plot_de_rational, c_plot_de_rational , d_plot_de_rational = de_func.x[0], de_func.x[1],de_func.x[2],de_func.x[3]
a_plot_sa_rational, b_plot_sa_rational, c_plot_sa_rational , d_plot_sa_rational = sa_func.x[0], sa_func.x[1],sa_func.x[2],sa_func.x[3]
 
# plot 
n_range = list(range(0,1001,1))
x_val = [3*k/1000 for k in n_range]
plt.scatter(x_val, y_val, label='data')
#plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [rational(x, a_plot_nm_rational, b_plot_nm_rational, c_plot_nm_rational , d_plot_nm_rational) for x in x_val], label='Nelder-Mead', color = 'red')
plt.plot(x_val, [rational(x, a_plot_lm_rational, b_plot_lm_rational, c_plot_lm_rational , d_plot_lm_rational ) for x in x_val], label='LM', color = 'green')
plt.plot(x_val, [rational(x, a_plot_de_rational, b_plot_de_rational, c_plot_de_rational , d_plot_de_rational ) for x in x_val], label='Differential Evolution ', color = 'blue')
plt.plot(x_val, [rational(x, a_plot_sa_rational, b_plot_sa_rational, c_plot_sa_rational , d_plot_sa_rational ) for x in x_val], label='Simulated Annealing, ', color = 'purple')

# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Rational approximation, stochastic and metaheuristic algorithms')
plt.rcParams["figure.figsize"]=(10,10)

plt.legend()

#Number 2

pip install utils

import utils

# read the data 
citys=pd.read_table('/content/wg22_xy.txt',header=None)
citys.columns=['x']
citys['y']=None
citys.drop(citys.head(2).index, inplace=True)

# some needed manipulation with data 
for i in range(2,len(citys)):
    coordinate=citys['x'][i].split()
    citys['x'][i]=float(coordinate[0])
    citys['y'][i]=float(coordinate[1])
    
citys=citys.drop([22])
citys=citys.drop([23])

start=list(citys.iloc[0])
end=list(citys.iloc[0])
citys.index=[i for i in range(len(citys))]

#matrix of distances
matrix = pd.DataFrame(distance_matrix(citys.values, citys.values), index=citys.index, columns=citys.index)

citys_copy_new = citys.append({'x' : end[0],
                    'y' : end[1]},ignore_index=True)
                    

citys=citys.drop([0])
paths=[i+1 for i in range(len(citys))] # initiate path
def CalDistance(x,y):
    return math.sqrt(x**2+y**2)
    
#here we calculate our lenght of paths
def CalLength(matrix, paths):
    length=0
    n=1 
    for i in range(len(paths)):
        if i==0:
            length+=matrix[0][paths[i]]
            n+=1
        elif n<len(paths):
            length+=matrix[paths[i]][paths[i+1]]
            n+=1
        else:
            length+=matrix[0][paths[i]]
    return length


distance1=0
distance2=0
dif=0
for i in range(len(citys)):  
    #np.random.shuffle(path)
    newPaths1= list(np.random.permutation(paths))
    newPaths2=list(np.random.permutation(paths))
    distance1= CalLength(matrix, newPaths1)
    #distance2= CalLength(citys,newPaths2,start,end)
    distance2= CalLength(matrix, newPaths2)
    difNew=abs(distance1-distance2)
    if difNew>=dif:
        dif=difNew
#out algorithm
Pr=0.5 #initiate accept possibility
T0=dif/Pr#initiate terperature

T=T0
Tmin=T/50
k=10*len(paths) #times of internal circulation 
length=0#initiate distance according to the initiate path
#length= CalLength(citys,paths,start,end)
length= CalLength(matrix,paths)
print(length)
t=0 #time 

initialPath=list(np.random.permutation(paths))
#length=CalLength(citys,initialPath,start,end)
length= CalLength(matrix,paths)
optimalPath = initialPath.copy()
optimalLength=length
t=0
while T>Tmin:
    for i in range(k):
        newPaths=optimalPath.copy()
        for j in range(int(T0/50)):
            a=0
            b=0
            while a==b:
                a=np.random.randint(0,len(paths))
                b=np.random.randint(0,len(paths))
            te=newPaths[a]
            newPaths[a]=newPaths[b]
            newPaths[b]=te
        #newLength=CalLength(citys,newPaths,start,end)
        newLength=CalLength(matrix,newPaths)
        if newLength<optimalLength:
            optimalLength=newLength
            optimalPath=newPaths
        else:
             #metropolis principle
             p=math.exp(-(newLength-optimalLength)/T)
             r=np.random.uniform(low=0,high=1)
             if r<p:
                 optimalLength=newLength
                 optimalPath=newPaths

    back=np.random.uniform(low=0,high=1)
    if back>=0.85:
        T=T*2
        continue
    t+=1
    print (t)
    T=T0/(1+t)
print (optimalLength)

initialPath = [0] + initialPath  +[20]
optimalPath = [0] + optimalPath + [20]

#plot initial situation
citys_copy_new['order']=initialPath
citys_order=citys_copy_new.sort_values(by=['order'])
plt.plot(citys_order['x'],citys_order['y'],  marker='o', linestyle='dashed')
plt.plot(citys_order['x'][0],citys_order['y'][0], 'ro')
plt.text(citys_order['x'][0],citys_order['y'][0],'First')
plt.title('Initial situation')
plt.xlabel('x', color='gray')
plt.ylabel('y',color='gray')
plt.show()

#plot optimal situation
citys_copy_new['order']=optimalPath
citys_order=citys_copy_new.sort_values(by=['order'])

plt.plot(citys_order['x'],citys_order['y'],  marker='o', linestyle='dashed')
plt.plot(citys_order['x'][0],citys_order['y'][0], 'ro')
plt.text(citys_order['x'][0],citys_order['y'][0],'First')
plt.title('Optimal situation')
plt.xlabel('x', color='gray')
plt.ylabel('y',color='gray')
plt.show()

#compare lenght
print('non-optimal Length '+ ' ' + str(length))
print('optimal Length '+ ' ' + str(optimalLength))