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
res_rational = scipy.optimize.minimize(lambda a : D(a[0],a[1], a[2], a[3], rational, y_val), [-2,1,1,-2], method = 'Nelder-Mead', options={'fatol': 0.01})
res_rational

#algorithm LM
lm_func = scipy.optimize.root(lambda a : D_copy(a[0],a[1], a[2], a[3], rational, y_val), [-2,1,1,-2], method = 'lm', options={'ftol': 0.001})
lm_func

#algorithm Differential Evolution
de_func = scipy.optimize.differential_evolution(lambda a : D(a[0],a[1], a[2], a[3], rational, y_val), [(-3,3),(-3,3),(-3,3),(-3,3)], strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7)
de_func

#algorithm Annealing
sa_func = scipy.optimize.dual_annealing(lambda a : D(a[0],a[1], a[2], a[3], rational, y_val), [(-3,3),(-3,3),(-3,3),(-3,3)])
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
# Modules
import math
import numpy as np
import matplotlib.pyplot as plt

# calulate length between two points

def length (n1,n2):
	return math.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)


# calculate total length to traverse all points

def total_length(arr,n):
	l=length(arr[0],arr[n-1])
	for i in range(n-1):
		l+= length(arr[i],arr[i+1])
	return l
		

	
	
# two_opt optimization for simulated annealing, using a random probabilty function to do selection

def two_opt_optimization(sol_arr,t,n):
	
	# picking two pair of consecutive integers, making sure they are not same
	ai =np.random.randint(0,n-1)
	bi =(ai+1)%n 
	ci =np.random.randint(0,n-1)
	di =(ci+1)%n
	
	if ai != ci and bi != ci:
		a =sol_arr[ai]
		b =sol_arr[bi]
		c =sol_arr[ci]
		d =sol_arr[di]
		
		# old lengths
		ab =length(a,b)
		cd =length(c,d)
		# new lengths, if accepted by our probability function
		ac =length(a,c)
		bd =length(b,d)
		
		diff = ( ab + cd ) - ( ac + bd )
		
		p = 0
		# for negative diff-> we'll use boltzman probabilty distribution equation-> P(E)=exp(-E/kT)
		if diff < 0:
			# k is considered to be 1
			p = math.exp( diff/t )
			
		# we'll sometimes skip the good solution
		elif diff > 0.05 :
			p = 1
			
		#print p	
		if(np.random.random() < p ):
			
			new_arr = list(range(0,n))
			new_arr[0]=sol_arr[ai]
			i = 1
			
			while bi!= ci:
				
				new_arr[i]=sol_arr[ci]
				i = i+1
				ci = (ci-1)%n
				
			new_arr[i]=sol_arr[bi]
			i = i+1
			
			while ai!= di:
				new_arr[i] =sol_arr[di]
				i = i+1
				di =(di+1)%n
				
				
			# animate this frame	
			#animate()
			
			return new_arr
			
	return sol_arr
				
				
# Simmulated Annealing algorithm----------------------------------------------	
	
def sa_algorithm (input_data):
	
	#length of input_data
	n=len(input_data)
	
	#creating a base solution
	sol_arr=input_data
	print("Initial order")
	print(sol_arr)
	
	#plot initial solution
	#plt.axis([-100,1100,-100,1100])
	#plt.plot(input_data[:,0],input_data[:,1],'ro')
	
	#initial temperature
	t = 100
	
	#current length
	min_l=total_length(sol_arr,n)
	
	i=0
	best_arr=[]
	
	while t>0.1:
		
		i= i+1
		
		#two_opt method- for optimization
		sol_arr=two_opt_optimization(sol_arr,t,n)
		
		#after 200 steps restart the process until the temperature is less than 0.1
		if i>=200 :
				
			i=0
			current_l=total_length(sol_arr,n)
			
			#because input size is approx. 200 i'm keeping the cooling schedule slow
			t = t*0.9995
			#print t
			
			if current_l < min_l:
				print(current_l)
				min_l=current_l
				best_arr=sol_arr[:]
	
	return best_arr
				
			 


#global variables
input_data = []




s="new.txt" #File new.txt contains city data
n=0
	
with open(s) as f:
	for line in f:
		numbers_str = line.split('.')
		x=int(numbers_str[0])
		y=int(numbers_str[1])
		input_data.append((x,y))
			
		
final_arr = sa_algorithm(input_data)
final_l = total_length(final_arr,n)

origin = initial_order[0]
initial_order = [(-57, 28), (54, -65), (46, 79), (8, 111), (-36, 52), (-22, -76), (34, 129), (74, 6), (-6, -41), (21, 45), (37, 155), (-38, 35), (-5, -24), (70, -74), (59, -26), (114, -56), (83, -41), (-40, -28), (21, -12), (0, 71)]
initial_order.append(origin)

plt.plot([x[0] for x in initial_order],[y[1] for y in initial_order],  marker='o', linestyle='dashed')
plt.plot(initial_order[0][0],initial_order[0][1], 'ro')
plt.text(initial_order[0][0],initial_order[0][1], 'First')
plt.title('Initial situation')
plt.xlabel('x', color='gray')
plt.ylabel('y',color='gray')
plt.show()

citys_order=final_arr
final_arr = final_arr[final_arr.index(origin):] + final_arr[:final_arr.index(origin)+1]
plt.plot([x[0] for x in final_arr],[y[1] for y in final_arr],  marker='o', linestyle='dashed')
plt.plot(final_arr[0][0],final_arr[0][1], 'ro')
plt.text(final_arr[0][0],final_arr[0][1], 'First')
plt.title('Final situation')
plt.xlabel('x', color='gray')
plt.ylabel('y',color='gray')
plt.show()
