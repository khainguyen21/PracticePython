### Lesson 1

# # Ex1: Write a program to count positive and negative numbers in a list
# data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]
#
# # Initialize counters for positive and negative numbers
# positive = 0
# negative = 0

# # Loop through each number in a list
# for number in data1:
#
#     # Increment positive counter by one if current number is positive
#     if number > 0:
#         positive += 1
#
#     # Otherwise, increment negative counter by one
#     else:
#         negative += 1
#
# # Display total positive and negative number in a list
# print("Total positive number in a list is ", positive)
# print("Total negative number in a list is ", negative)

# # Ex2: Given a list, extract all elements whose frequency is greater than k.
# data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
# k = 3
#
# # Initialize the dictionary to store frequency of a number in the data2 list
# frequency = {}
#
# # Initialize the list to store all numbers in the list with frequency higher than k
# results = []
#
# # Loop through the data2 list
# for num in data2:
#
#     # Check if num in already exists in the dictionary as a key
#     if num in frequency:
#
#         # Increment the value of a key by 1
#         frequency[num] += 1
#
#     # Otherwise, initialize a new key and set value to 1
#     else:
#         frequency[num] = 1
#
# # Loop through the data2 list
# for num in data2:
#
#     # Check if value of the key is greater than k
#     if frequency[num] > k:
#
#         # Append to the result list
#         results.append(num)
#
# print(frequency)
# print(results)

# Ex3: find the strongest neighbour. Given an array of N positive integers.
# The task is to find the maximum for every adjacent pair in the array.
# data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]

# index1 = 0
# index2 = 1
# results = []
# while index2 < len(data3):
#     results.append(max(data3[index1] , data3[index2]))
#     index1 += 1
#     index2 += 1
#
# print(results)

# results = []
# # not include
# for i in range(len(data3)-1):
#     results.append(max(data3[i], data3[i+1]))
# print(results)

# Ex4: print all Possible Combinations from the three Digits
# data4 = [1, 2, 3]
#
# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#             if i != k and i != j and j != k:
#                 print(data4[i], data4[j], data4[k])


# Ex5: Given two matrices (2 nested lists), the task is to write a Python program
# to add elements to each row from initial matrix.
# For example: Input : test_list1 = [[4, 3, 5,], [1, 2, 3], [3, 7, 4]], test_list2 = [[1], [9], [8]]
# Output : [[4, 3, 5, 1], [1, 2, 3, 9], [3, 7, 4, 8]]
# data5_list1 = [[4, 3, 5, ], [1, 2, 3], [3, 7, 4]]
# data5_list2 = [[1, 3], [9, 3, 5, 7], [8]]
#
# for i in range(len(data5_list1)):
#     data5_list1[i].extend(data5_list2[i])
#
# print(data5_list1)
#

# Ex6:  Write a program which will find all such numbers which are divisible by 7
# but are not a multiple of 5, between 2000 and 3200 (both included).
# The numbers obtained should be printed in a comma-separated sequence on a single line.

# results = []
# for num in range(2000, 3201):
#     if num % 7 == 0 and num % 5 != 0:
#         results.append(str(num))
#
# print(", ".join(results))

# Ex7: Write a program, which will find all such numbers between 1000 and 3000 (both included) such that
# each digit of the number is an even number. The numbers obtained should be printed in a comma-separated
# sequence on a single line.

# even_numbers = []
# for num in range(1000, 3001):
#     if all(int(digit) % 2 == 0 for digit in str(num)):
#         even_numbers.append(num)
#
# print(even_numbers)

# # Ex8: Let user type 2 words in English as input. Print out the output
# # which is the shortest chain according to the following rules:
# # - Each word in the chain has at least 3 letters
# # - The 2 input words from user will be used as the first and the last words of the chain
# # - 2 last letters of 1 word will be the same as 2 first letters of the next word in the chain
# # - All the words are from the file wordsEn.txt
# # - If there are multiple shortest chains, return any of them is sufficient


### Lesson 2
import numpy as np

# Ex1: Write a NumPy program to reverse an array (first element becomes last).
# Input: [12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37]

# arr = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
#                 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
# print(np.flip(arr))
# print(arr[::-1])

# Ex2: Write a NumPy program to test whether each element of a 1-D array is also present in a second array
# Input Array1: [ 0 10 20 40 60]
#       Array2: [10, 30, 40]

# array1 = np.array([0, 10, 20, 40, 60])
# array2 = np.array([10, 30, 40])
# count = np.isin(array1, array2)
# print(count)
# #array1[result] selects only the elements where result is True.
# print(array1[count])

# # Ex3: Write a NumPy program to find the indices of the maximum and minimum values along the given axis of an array
# # Input Array
# input_array = np.array([1,6,4,8,9,-4,-2,11])
# maximum = np.argmax(input_array)
# minimum = np.argmin(input_array)
#
# print(f"Index of max value is {maximum}")
# print(f"Index of min value is {minimum}")


# Ex4: Read the entire file story.txt and write a program to print out top 100 words occur most
# frequently and their corresponding appearance. You could ignore all
# punction characters such as comma, dot, semicolon, ...
# Sample output:
# house: 453
# dog: 440
# people: 312
# ...

# from collections import Counter
# # Read in the text files
# readTextFile = open("story.txt", encoding='utf-8').read().lower()
# #print(readTextFile)
#
# # Split each words in the file with whitespace delimiter; this includes spaces, tabs, and newlines
# split_words = np.array(readTextFile.split())
# #print(split_words)
#
# # Count word frequencies
# word_counts = Counter(split_words)
#
# # Take top 10 frequencies of each word
# top_10_word_counts = word_counts.most_common(10)
# #print(type(top_10_word_counts[9]))
#
# for word, frequency in top_10_word_counts:
#     #word, frequency = item
#     # word = item[0]
#     # frequency = item[1]
#     print(f"{word}: {frequency}")