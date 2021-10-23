import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random

#number 1
#realize brute_forse method as in the lectures 
def brute_forse(func, a, b, eps):
  count_func_f = 0
  iterations = 0
  smallest_so_far = a
  min_func_value = func(a)
  n = math.ceil((b -a)/eps)
  for k in range(1, n+1):
    x = a + k*(b-a)/n
    fx = func(x)
    count_func_f += 1
    if fx<min_func_value:
      smallest_so_far = x
      min_func_value=fx
  return count_func_f, iterations, smallest_so_far
  
#realize dichotomy method as in the lectures 
def dihotomia(func, a, b, eps):
  count_f = 0
  iterations = 0
  while abs(a -b)> eps:
    x_first = (a+b -eps/2)/2
    x_second = (a+b +eps/2)/2
    if func(x_first)<= func(x_second):
      count_f +=2
      iterations += 1
      a = a 
      b = x_second 
    else:
      b = b
      a = x_first
      count_f +=2
      iterations += 1
  return count_f, iterations, (a+b)/2

#realize gold ratio method as in the lectures 
def gold(func, a, b, eps):
  count_f = 2
  iterations = 0
  x_first = a+ (b - a) * ((3 - math.sqrt(5))/2)
  x_second = b+ (b - a) * ((-3 + math.sqrt(5))/2)
  func_x_first = func(x_first)
  func_x_second = func(x_second)
  while abs(a -b)> eps:
    iterations+=1
    if func_x_first<= func_x_second:  
      a = a 
      b = x_second 
      x_second = x_first
      func_x_second = func_x_first
      x_first = a+ (b - a) * ((3 - math.sqrt(5))/2)
      func_x_first = func(x_first)
      count_f += 1
    else:
      b = b
      a = x_first
      x_first = x_second
      func_x_first = func_x_second
      x_second = b+ (b - a) * ((-3 + math.sqrt(5))/2)
      func_x_second = func(x_second)
      count_f += 1
  return  count_f, iterations, (a+b)/2

#question one 
brute_forse(lambda x: x**3, 0, 1, 0.001)
dihotomia(lambda x: x**3, 0, 1, 0.001)
gold(lambda x: x**3, 0, 1, 0.001)
brute_forse(lambda x: abs(x - 0.2), 0, 1, 0.001)
dihotomia(lambda x: abs(x - 0.2), 0, 1, 0.001)
gold(lambda x: abs(x - 0.2), 0, 1, 0.001)
brute_forse(lambda x: x*np.sin(1/x), 0.01, 1, 0.001)
dihotomia(lambda x: x*np.sin(1/x), 0.01, 1, 0.001)
gold(lambda x: x*np.sin(1/x), 0.01, 1, 0.001)

#extra example for number one
brute_forse(lambda x: 6*x*(x-1)*(x+1), -1.5, 1.5, 0.001)
dihotomia(lambda x: 6*x*(x-1)*(x+1), -1.5, 1.5, 0.001)
gold(lambda x: 6*x*(x-1)*(x+1), -1.5, 1.5, 0.001)


#number 2
alfa = random.random()
beta = random.random()
y_val = []
#real_val - data without the noise
real_val = []
for k in range(0,101):
  delta = random.gauss(0, 1)
  x = k/100
  y_val.append(alfa*x + beta + delta)
  real_val.append(alfa*x + beta)
  
# Gauss uses dichotomy to find minima of single parameter function
def dihotomia_for_gauss(func, a, b, eps):
  count_f = 0
  iterations = 0
  while abs(a -b)> eps:
    x_first = (a+b -eps/2)/2
    x_second = (a+b +eps/2)/2
    if func(x_first)<= func(x_second):
      count_f +=2
      iterations += 1
      a = a 
      b = x_second 
    else:
      count_f +=2
      b = b
      a = x_first
  return (a+b)/2, count_f


#square error function
def D(a,b,func, y):
  summa = 0
  for k in range(0, 101):
    summa += (func(k/100,a,b) - y[k])**2
  return summa 
  
#our approximation functions 
def linear(x, a, b):
  return a*x + b
  
def rational(x, a, b):
  return a/(1 + x*b)
  
# frute-forse for 2 - dim data
def brute_forse_two(func, eps):
  #x, y = 0, 0
  #smallest_x, smallest_y = 0, 0
  x, y = 0, 0
  smallest_x, smallest_y = 0, 0
  min_func_value = func(x, y)
  count_f = 1
  while y <= 1:
    while x <= 1:
      x += eps
      func_n = func(x, y)
      count_f += 1
      if min_func_value > func_n:
        smallest_x = x
        smallest_y = y
        min_func_value = func_n
    x = 0 #0
    y += eps
  return smallest_x, smallest_y, count_f
 
 
# realize Gauss for 2 - dim data
def gauss(func, eps):
  a,b = 0,1
  x_prev, y_prev = (b-a)/2, (b-a)/2
  fun_calls_x, fun_calls_y = 0, 0 
  count_f = 0
  x_curr, fun_calls_x = dihotomia_for_gauss(lambda x: func(x,y_prev), a, b, eps)
  y_curr, fun_calls_y = dihotomia_for_gauss(lambda y: func(x_curr,y), a, b, eps)
  count_f +=fun_calls_x
  count_f += fun_calls_y
  while abs(func(x_curr, y_curr)-func(x_prev, y_prev)) >= eps:
    x_prev, y_prev = x_curr, y_curr
    x_curr, fun_calls_x = dihotomia_for_gauss(lambda x: func(x,y_prev), a, b, eps)
    y_curr, fun_calls_y = dihotomia_for_gauss(lambda y: func(x_curr,y), a, b, eps)
    count_f += fun_calls_x
    count_f += fun_calls_y
  return x_curr, y_curr, count_f
  
#realize NM method
res_lin = scipy.optimize.minimize(lambda a : D(a[0],a[1],linear, y_val), [0,0], method = 'Nelder-Mead', bounds= [[0,1], [0,1]], options={'xatol': 0.001})
res_rational = scipy.optimize.minimize(lambda a : D(a[0],a[1],rational, y_val), [0,0], method = 'Nelder-Mead', bounds= [[0,1], [0,1]], options={'xatol': 0.001})

#estimated linear parameters
a_plot_brute_lin, b_plot_brute_lin, counter_lin_br = brute_forse_two(lambda a,b: D(a,b,linear, y_val),0.001)
a_plot_gauss_lin, b_plot_gauss_lin, counter_gauss_br  = gauss(lambda a,b: D(a,b,linear, y_val),0.1)
a_plot_nm_lin, b_plot_nm_lin = res_lin.x[0], res_lin.x[1]

#estimated rational parameters
a_plot_brute_rational, b_plot_brute_rational, counter_rat_br  = brute_forse_two(lambda a,b: D(a,b,rational, y_val),0.001)
a_plot_gauss_rational, b_plot_gauss_rational, counter_rat_gauss = gauss(lambda a,b: D(a,b,rational, y_val),0.1)
a_plot_nm_rational, b_plot_nm_rational = res_rational.x[0], res_rational.x[1]

#plot all methods for linear app
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [linear(x, a_plot_brute_lin, b_plot_brute_lin) for x in x_val], label='brute forse')
plt.plot(x_val, [linear(x, a_plot_gauss_lin, b_plot_gauss_lin) for x in x_val], label='Gauss')
plt.plot(x_val, [linear(x, a_plot_nm_lin, b_plot_nm_lin) for x in x_val], label='Nelder-Mead')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Linear approximation, all methods')
plt.legend()
plt.show()

#plot one search for linear app
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [linear(x, a_plot_brute_lin, b_plot_brute_lin) for x in x_val], label='brute forse')
#plt.plot(x_val, [linear(x, a_plot_gauss_lin, b_plot_gauss_lin) for x in x_val], label='Gauss')
#plt.plot(x_val, [linear(x, a_plot_nm_lin, b_plot_nm_lin) for x in x_val], label='Nelder-Mead')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Linear approximation, Exhaustive search')
plt.legend()
plt.show()

#plot 2 for linear 
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [linear(x, a_plot_brute_lin, b_plot_brute_lin) for x in x_val], label='brute forse')
plt.plot(x_val, [linear(x, a_plot_gauss_lin, b_plot_gauss_lin) for x in x_val], label='Gauss')
#plt.plot(x_val, [linear(x, a_plot_nm_lin, b_plot_nm_lin) for x in x_val], label='Nelder-Mead')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Linear approximation, Exhaustive search and Gauss methods')
plt.legend()
plt.show()

#plot all for rational 
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, [rational(x, a_plot_brute_rational, b_plot_brute_rational) for x in x_val], label='brute forse')
plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [rational(x, a_plot_gauss_rational, b_plot_gauss_rational) for x in x_val], label='Gauss')
plt.plot(x_val, [rational(x, a_plot_nm_rational, b_plot_nm_rational) for x in x_val], label='Nelder-Mead')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Rational approximation, all methods')
plt.legend()
plt.show()

#plot brite force for rational 
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, [rational(x, a_plot_brute_rational, b_plot_brute_rational) for x in x_val], label='brute forse')
plt.plot(x_val, real_val, label='generating function', color = 'black')
#plt.plot(x_val, [rational(x, a_plot_gauss_rational, b_plot_gauss_rational) for x in x_val], label='Gauss')
#plt.plot(x_val, [rational(x, a_plot_nm_rational, b_plot_nm_rational) for x in x_val], label='Nelder-Mead')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Rational approximation, Exhaustive search')
plt.legend()
plt.show()

#plot 2 for rational 
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, [rational(x, a_plot_brute_rational, b_plot_brute_rational) for x in x_val], label='brute forse')
plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [rational(x, a_plot_gauss_rational, b_plot_gauss_rational) for x in x_val], label='Gauss')
#plt.plot(x_val, [rational(x, a_plot_nm_rational, b_plot_nm_rational) for x in x_val], label='Nelder-Mead')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Rational approximation, Exhaustive search and Gauss methods')
plt.legend()
plt.show()
