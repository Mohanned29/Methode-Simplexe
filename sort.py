import matplotlib.pyplot as plt
import numpy as np
import random
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def time_sorting_algorithm(sort_func, arr):
    start_time = time.time()
    sort_func(arr.copy())
    return time.time() - start_time

# Generate input sizes
input_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000]

# Sorting algorithms to compare
sorting_algorithms = {
    'Bubble Sort': bubble_sort,
    'Quick Sort': quick_sort,
    'Merge Sort': merge_sort
}

# Measure sorting times
sorting_times = {name: [] for name in sorting_algorithms}

for size in input_sizes:
    for name, algorithm in sorting_algorithms.items():
        # Generate random array for each test
        arr = [random.randint(0, 10000) for _ in range(size)]
        execution_time = time_sorting_algorithm(algorithm, arr)
        sorting_times[name].append(execution_time)

# Plotting
plt.figure(figsize=(10, 6))
for name, times in sorting_times.items():
    plt.plot(input_sizes, times, marker='o', label=name)

plt.title('Sorting Algorithm Performance Comparison')
plt.xlabel('Input Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
