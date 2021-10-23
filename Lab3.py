import scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random

#our data
alfa = random.random()
beta = random.random()
y_val = []
real_val = []
for k in range(0,101):
  x = k/100
  delta = random.gauss(0,1)
  y_val.append(alfa*x + beta + delta)
  real_val.append(alfa*x + beta)

  
def D(a,b,func, y):
  summa = 0
  for k in range(0, 101):
    summa += (func(k/100,a,b) - y[k])**2
  return summa

def D_copy(a,b,func, y):
  summa = 0
  for k in range(0, 101):
    summa += (func(k/100,a,b) - y[k])**2
  return [summa,summa]

def linear(x, a, b):
  return a*x + b

def rational(x, a, b):
  return a/(1 + x*b)

#gradient for linear function
def grad_lin(a):
  summa_1 = 0
  summa_2 = 0
  for k in range(1,101):
    x = k/100
    summa_1 += (a[0]*x + a[1] -y_val[k])*x*2
    summa_2 += (a[0]*x + a[1] -y_val[k])*2
  return np.array([summa_1, summa_2])

#gradient for rational function
def grad_rat(a):
  summa_1 = 0
  summa_2 = 0
  for k in range(0,101):
    x = k/100
    summa_1 += (a[0]/(1 + a[1]*x) -y_val[k])*2*(1/(1 + a[1]*x))
    summa_2 += -(a[0]/(1 + a[1]*x) -y_val[k])*2*x*a[0]/(1 + a[1])**2
  return np.array([summa_1, summa_2])

def gradient_descent(
    gradient, start, learn_rate, n_iter=20000, tolerance=1e-09
):
    vector = start
    iteration = 0
    for iteration in (range(n_iter)):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            print("Criterion stop, iterations = ", iteration)
            break
        vector += diff
    return vector
  
a_plot_gd_lin, b_plot_gd_lin = gradient_descent(grad_lin,[random.random(),random.random()], 0.001)
a_plot_gd_rat, b_plot_gd_rat = gradient_descent(grad_rat,[0.5,-0.6], 0.001)

#inverse hessian for linear function
def inverse_hess(a):
  summa_1 = 0
  summa_2 = 0
  summa_3 = 0
  summa_4 = 0
  for k in range(0,101):
    x = k/100
    summa_1 += x*2*x 
    summa_2 += 2*x 
    summa_3 += 2*x 
    summa_4 += 2
  A_mn = summa_1*summa_4 -summa_2*summa_3
  return np.array([[summa_4/A_mn, -summa_2/A_mn], [-summa_3/A_mn, summa_1/A_mn]])

# hessian for linear function
def lin_hess(a):
  summa_1 = 0
  summa_2 = 0
  summa_3 = 0
  summa_4 = 0
  for k in range(0,101):
    x = k/100
    summa_1 += x*2*x 
    summa_2 += 2*x 
    summa_3 += 2*x 
    summa_4 += 2
  return np.array([[summa_1, summa_2], [summa_3, summa_4]])

#inverse hessian for rational function
def inverse_hess_rat(a):
  summa_1 = 0
  summa_2 = 0
  summa_3 = 0
  summa_4 = 0
  for k in range(0,101):
    x = k/100
    summa_1 += 2/((1 + a[1]*x)**2)
    summa_2 += -2*a[0]*x*(1/(1 + a[1]*x)**3) - 2*x*(1/(1 + a[1]*x)**2)* (a[0]/(1 + a[1]*x) -y_val[k])
    summa_3 += -2*a[0]*x*(1/(1 + a[1]*x)**3) - 2*x*(1/(1 + a[1]*x)**2)* (a[0]/(1 + a[1]*x) -y_val[k])
    summa_4 += 2*(a[0]**2)*(x**2)*(1/(1 + a[1]*x)**4) + 4*(x**2)*a[0]*(1/(1 + a[1]*x)**3)* (a[0]/(1 + a[1]*x) -y_val[k])
  A_mn = summa_1*summa_4 -summa_2*summa_3
  return np.array([[summa_4/A_mn, -summa_2/A_mn], [-summa_3/A_mn, summa_1/A_mn]])

# hessian for rational function
def hess_rat(a):
  summa_1 = 0
  summa_2 = 0
  summa_3 = 0
  summa_4 = 0
  for k in range(0,101):
    x = k/100
    summa_1 += 2/((1 + a[1]*x)**2)
    summa_2 += -2*a[0]*x*(1/(1 + a[1]*x)**3) - 2*x*(1/(1 + a[1]*x)**2)* (a[0]/(1 + a[1]*x) -y_val[k])
    summa_3 += -2*a[0]*x*(1/(1 + a[1]*x)**3) - 2*x*(1/(1 + a[1]*x)**2)* (a[0]/(1 + a[1]*x) -y_val[k])
    summa_4 += 2*(a[0]**2)*(x**2)*(1/(1 + a[1]*x)**4) + 4*(x**2)*a[0]*(1/(1 + a[1]*x)**3)* (a[0]/(1 + a[1]*x) -y_val[k])
  return np.array([[summa_1, summa_2], [summa_3, summa_4]])

lin_CG = scipy.optimize.minimize(lambda a : D(a[0],a[1],linear, y_val), (random.random(),random.random()), method = 'CG', options={'gtol': 0.001})
rat_CG = scipy.optimize.minimize(lambda a : D(a[0],a[1],rational, y_val), (random.random(),random.random()), method = 'CG', options={'gtol': 0.001})

a_plot_cg_lin, b_plot_cg_lin = lin_CG.x[0], lin_CG.x[1]
a_plot_cg_rat, b_plot_cg_rat = rat_CG.x[0], rat_CG.x[1]

lin_newton = scipy.optimize.minimize(lambda a : D(a[0],a[1],linear, y_val), [0,0], method = 'Newton-CG', jac=grad_lin, hess = lin_hess,  bounds= [[-1,1], [-1,1]], options={'xtol': 0.0000000000005})
rat_newton = scipy.optimize.minimize(lambda a : D(a[0],a[1],rational, y_val), [0,0], method = 'Newton-CG', jac=grad_rat, hess = hess_rat,  bounds= [[-1,1], [-1,1]], options={'xtol': 0.0000000000005})

a_plot_newton_lin, b_plot_newton_lin = lin_newton.x[0], lin_newton.x[1]
a_plot_newton_rat, b_plot_newton_rat = rat_newton.x[0], rat_newton.x[1]


lin_lm = scipy.optimize.root(lambda a : D_copy(a[0], a[1],linear, y_val), [random.random(),random.random()], method='lm',  options={'ftol': 0.001})
rat_lm = scipy.optimize.root(lambda a : D_copy(a[0], a[1],rational, y_val), [random.random(),random.random()], method='lm',options={'ftol': 0.001} )

a_plot_lm_lin, b_plot_lm_lin = lin_lm.x[0], lin_lm.x[1]
a_plot_lm_rat, b_plot_lm_rat = rat_lm.x[0], rat_lm.x[1]


#plot linear functions
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [linear(x, a_plot_gd_lin, b_plot_gd_lin) for x in x_val], label='gradient descent', color = 'green')
plt.plot(x_val, [linear(x, a_plot_newton_lin, b_plot_newton_lin) for x in x_val], label='Newton')
plt.plot(x_val, [linear(x, a_plot_cg_lin, b_plot_cg_lin) for x in x_val], label='CG', color = 'red')
plt.plot(x_val, [linear(x, a_plot_lm_lin, b_plot_lm_lin) for x in x_val], label='LM', color = 'purple')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Linear approximation')
plt.legend()
plt.show()

#plot rational functions
n_range = list(range(0,101,1))
x_val = [k/100 for k in n_range]
plt.scatter(x_val, y_val, label='data')
plt.plot(x_val, real_val, label='generating function', color = 'black')
plt.plot(x_val, [rational(x, a_plot_gd_rat, b_plot_gd_rat) for x in x_val], label='gradient descent', color = 'green')
plt.plot(x_val, [rational(x, a_plot_newton_rat, b_plot_newton_rat) for x in x_val], label='Newton')
plt.plot(x_val, [rational(x, a_plot_cg_rat, b_plot_cg_rat) for x in x_val], label='CG', color = 'red')
plt.plot(x_val, [rational(x, a_plot_lm_rat, b_plot_lm_rat) for x in x_val], label='LM', color = 'purple')
# naming the x axis
plt.xlabel('')
# naming the y axis
plt.ylabel('')
plt.title('Rational approximation')
plt.legend()
plt.show()

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

# frute-forse for 2 - dim data
def brute_forse_two(func, eps):
  #x, y = 0, 0
  #smallest_x, smallest_y = 0, 0
  x, y = -1+eps, -1+eps
  smallest_x, smallest_y = -1+eps, -1+eps
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
    x = -1+eps #0
    y += eps
  return smallest_x, smallest_y, count_f

# realize Gauss for 2 - dim data
def gauss(func, eps):
  a,b = -1,1
  #x_start, y_start = (b-a)/2, (b-a)/2
  x_start, y_start = random.random(), random.random()
  x_prev, y_prev = x_start, y_start
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

res_lin = scipy.optimize.minimize(lambda a : D(a[0],a[1],linear, y_val), [0,0], method = 'Nelder-Mead', bounds= [[0,1], [0,1]], options={'xatol': 0.001})
res_rational = scipy.optimize.minimize(lambda a : D(a[0],a[1],rational, y_val), [0,0], method = 'Nelder-Mead', bounds= [[0,1], [0,1]], options={'fatol': 0.001})

#estimated linear parameters
a_plot_nm_lin, b_plot_nm_lin = res_lin.x[0], res_lin.x[1]
a_plot_gauss_lin, b_plot_gauss_lin, counter_gauss_br  = gauss(lambda a,b: D(a,b,linear, y_val),0.001)
a_plot_brute_lin, b_plot_brute_lin, counter_lin_br = brute_forse_two(lambda a,b: D(a,b,linear, y_val),0.001)

#estimated rational parameters
a_plot_brute_rational, b_plot_brute_rational, counter_rat_br  = brute_forse_two(lambda a,b: D(a,b,rational, y_val),0.001)
a_plot_gauss_rational, b_plot_gauss_rational, counter_rat_gauss = gauss(lambda a,b: D(a,b,rational, y_val),0.001)
a_plot_nm_rational, b_plot_nm_rational = res_rational.x[0], res_rational.x[1]



