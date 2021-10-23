import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt

#constant found experimentally
def constant(v):
  return 0.25/(10**5)
average = []
n_range = list(range(1,2001))
for n in n_range:
  time_raw = []
  for a in range(1,6):
    x = abs(np.random.randn(1,n))
    start_time = time.perf_counter()
    constant(x)
    time_raw.append((time.perf_counter() - start_time))
  average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [constant(x) for x in n_range], label='theoretical')

# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of the constant function')
plt.legend()
plt.show()

def summa(v):
  return sum(v)

def linear(v):
  return v*0.0002/2000
average = []
n_range = list(range(1,2001))
for n in n_range:
    time_raw = []
    for a in range(1, 6):
        x = abs(np.random.randn(1,n))
        x = list(x[0])
        tart_time = time.perf_counter()
        summa(x)
        time_raw.append((time.perf_counter() - tart_time))
    average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [linear(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of the sum function')
plt.legend()
plt.show()

def prod(v):
    return np.prod(v, axis=1)

def linear(v):
  return v*0.28/(10**8) + 0.000006
average = []
n_range = list(range(1,2001))
for n in n_range:
    time_raw = []
    for a in range(1, 6):
        x = abs(np.random.randn(1,n))
        tart_time = time.perf_counter()
        prod(x)
        time_raw.append((time.perf_counter() - tart_time))
    average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [linear(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of the product function')
plt.legend()
plt.show()

# realise naive poly by this 
def poly_naive(x, coeff):
    result = coeff[0]
    for i in range(1, len(coeff)):
      counter = i
      x_i = 1
      #compute x to the power i naivly 
      while counter != 0:
        x_i *= x
        counter = counter-1
      result = result + coeff[i] * x_i
    return result
    
def horner(poly, n, x):
    # Initialize result
    result = poly[0] 
    # Evaluate value of polynomial
    # using Horner's method
    for i in range(n-1, 0, -1):
        result = result*x + poly[i]
    return result

def square(v):
  return (v**2)*0.2/(4*10**6) + 0.000006
average = []
n_range = list(range(1,2001))
for n in n_range:
    time_raw = []
    for a in range(1, 6):
        x = abs(np.random.randn(1,n))
        x=list(x[0])
        tart_time = time.perf_counter()
        poly_naive(1.5, x)
        time_raw.append((time.perf_counter() - tart_time))
    average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [square(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of the naive polynomial')
plt.legend()
plt.show()

def linear(v):
  return v*0.0003/(10**3) + 0.000006
average = []
n_range = list(range(1,2001))
for n in n_range:
    time_raw = []
    for a in range(1, 6):
        x = abs(np.random.randn(1,n))
        x=list(x[0])
        tart_time = time.perf_counter()
        horner(x, len(x), 1.5)
        time_raw.append((time.perf_counter() - tart_time))
    average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [linear(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of the Horner method')
plt.legend()
plt.show()

def bSort(array):
    # determine the length of the array
    length = len(array)
    #Outer loop, number of passes N-1
    for i in range(length):
        # Inner loop, N-i-1 passes
        for j in range(0, length-i-1):
            #Swapping elements in places
            if array[j] > array[j+1]:
                temp = array[j]
                array[j] = array[j+1]
                array[j+1] = temp
       
       
def square(v):
  return (v**2)*0.46/(4*10**6) + 0.000006
average = []
n_range = list(range(1,2001, 1))
for n in n_range:
    time_raw = []
    for a in range(1, 6):
        x = abs(np.random.randn(1,n))
        #transform numpy into python list
        x=list(x[0])
        tart_time = time.perf_counter()
        bSort(x)
        #print(summa(x))
        time_raw.append((time.perf_counter() - tart_time))
    average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [square(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of bubble sort')
plt.legend()
plt.show()

def quicksort(nums):
   if len(nums) <= 1:
       return nums
   else:
       q = random.choice(nums)
   l_nums = [n for n in nums if n < q]
 
   e_nums = [q] * nums.count(q)
   b_nums = [n for n in nums if n > q]
   return quicksort(l_nums) + e_nums + quicksort(b_nums)
def nlogn(v):
  return v*math.log(v)*0.00713/15201.8
import time
average = []
n_range = list(range(1,2001))
for n in n_range:
    time_raw = []
    for a in range(1, 6):
        x = abs(np.random.randn(1,n))
        x=list(x[0])
        tart_time = time.perf_counter()
        quicksort(x)
        time_raw.append((time.perf_counter() - tart_time))
    average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [nlogn(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of quicksort')
plt.legend()
plt.show()

# Python3 program to perform basic timSort
MIN_MERGE = 32
 
def calcMinRun(n):
    """Returns the minimum length of a
    run from 23 - 64 so that
    the len(array)/minrun is less than or
    equal to a power of 2.
 
    e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
    ..., 127=>64, 128=>32, ...
    """
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r
 
 
# This function sorts array from left index to
# to right index which is of size atmost RUN
def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1
 
 
# Merge function merges the sorted runs
def merge(arr, l, m, r):
     
    # original array is broken in two parts
    # left and right array
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])
 
    i, j, k = 0, 0, l
     
    # after comparing, we merge those two array
    # in larger sub array
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
 
        else:
            arr[k] = right[j]
            j += 1
 
        k += 1
 
    # Copy remaining elements of left, if any
    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1
 
    # Copy remaining element of right, if any
    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1
 
 
# Iterative Timsort function to sort the
# array[0...n-1] (similar to merge sort)
def timSort(arr):
    n = len(arr)
    minRun = calcMinRun(n)
     
    # Sort individual subarrays of size RUN
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)
 
    # Start merging from size RUN (or 32). It will merge
    # to form size 64, then 128, 256 and so on ....
    size = minRun
    while size < n:
         
        # Pick starting point of left sub array. We
        # are going to merge arr[left..left+size-1]
        # and arr[left+size, left+2*size-1]
        # After every merge, we increase left by 2*size
        for left in range(0, n, 2 * size):
 
            # Find ending point of left sub array
            # mid+1 is starting point of right sub array
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
 
            # Merge sub array arr[left.....mid] &
            # arr[mid+1....right]
            if mid < right:
                merge(arr, left, mid, right)
 
        size = 2 * size
        
def nlogn(v):
  return v*math.log(v)*0.011/15201.8
import time
average = []
n_range = list(range(1,2001, 1))
for n in n_range:
    time_raw = []
    for a in range(1, 6):
        x = abs(np.random.randn(1,n))
        x=list(x[0])
        tart_time = time.perf_counter()
        timSort(x)
        time_raw.append((time.perf_counter() - tart_time))
    average.append(sum(time_raw)/5)
plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [nlogn(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
 
# giving a title to my graph
plt.title('Time complexity of timsort')
plt.legend()
plt.show()

#number2
rng = np.random.default_rng(0)
def matmult(a,b):
    zip_b = zip(*b)
    #zip_b=b
    # uncomment next line if python 3 : 
    #zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]
          
def cubik(v):
  return 1.63*(v**3)/(8*10**9)+v**2/(10**7)
average = []
n_range = list(range(1,500,1))
for n in n_range:
  time_raw = []
  for a in range(1,6):
    A= rng.random((n, n))
    B =rng.random((n, n))
    start_time = time.perf_counter()
    matmult(A,B)
    time_raw.append((time.perf_counter() - start_time))
  average.append(sum(time_raw)/5)

plt.plot(n_range, average, label='experimental')
plt.plot(n_range, [cubik(x) for x in n_range], label='theoretical')
# naming the x axis
plt.xlabel('n')
# naming the y axis
plt.ylabel('Average time')
plt.title('Time complexity of product matrices A and B')

plt.legend()          
plt.show()
