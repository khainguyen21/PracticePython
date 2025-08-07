# # Lesson 1
#
# # Ex1: Write a program to count positive and negative numbers in a list
# data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]
#
# positive_array = [postive_num for postive_num in data1 if postive_num > 0]
#
# # Display total positive and negative number in a list
# print("Total positive number in a list is ", len(positive_array))
# print("Total negative number in a list is ", len(data1) - len(positive_array))

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


# 25, 50, 100 or 200. Số càng lớn thì càng chính xác, nhưng chạy càng lâu các bạn nhé
# model = api.load("glove-twitter-25")
#
# vec1 = model["man"]
# vec2 = model["woman"]
# #def calculate_similarity (word1, word2):
# dot_product = sum(word1 * word2 for word1, word2 in zip(vec1, vec2))
#
# magnitude_man = np.sqrt(sum(a**2 for a in vec1))
# magnitude_woman = np.sqrt(sum(b**2 for b in vec2))
#
# similarity = dot_product / (magnitude_woman * magnitude_man)
# print(similarity)
#
# result = model.similarity("man", "woman")
# print(result)








# print("1----------")
# result = model.most_similar(word, topn=10)
# print(result)

# print("2----------")
# result = model.most_similar(positive=["january", "february"], topn=10)
# print(result)
#
# print("3----------")
# result = model.similarity("man", "woman")
# print(result)
#
# print("4----------")
# result = model.most_similar(positive=["woman", "king"], negative=["man"], topn=1)
# print(result)
#
# print("5----------")
# result = model.most_similar(positive=["berlin", "vietnam"], negative=["hanoi"], topn=1)
# print(result)
#
# print("6----------")
# result = model.similarity("marriage", "happiness")
# print(result)

#
# TODO: Các bạn thử viết 2 cách khác nhau để tính cosine similarity
# giữa 2 vector nhé. Kết quả giống với khi dùng model.similarity() là được
# 1 cách là dùng numpy, 1 cách là không dùng numpy nhé


# Python for coding interviews

arr = [[0] * 4 for i in range(2)]
print(arr)

array = [0] * 4
print(array)

# String are similar to arrays
s = "abc"
print(s[0:2])

# However they are immutable, we cannot do code below
#s[0] = 'A'

# So this creates a new string, anytime we modify the string, it's considered O(N) time operation
s += "def"
print(s)

# Valid numeric strings can be converted
print(int("123") + int ("123")) # Output: 246

# And numbers can be converted to strings
print(str(123) + str(123)) # Output: 123123

# In rare, you may need the ASCII value of a char
print(ord("a"))
print(ord("b"))

# Combine a list of string (with an empty string delimiter)
strings = ["ab", "cd", "ef"]
print("".join(strings)) # Output: abcdef
print(" ".join(strings)) # Output: ab cd ef
print(", ".join(strings)) # Output: ab, cd, ef


# Queues (double ended queue) aka deque
from collections import deque

queue = deque()
queue.append(1)
queue.append(2)
queue.append(3)
queue.append(4)
queue.append(5)
print(queue) # deque([1, 2, 3, 4, 5])

queue.pop()
print(queue) # deque([1, 2, 3, 4])

queue.popleft() # O(1)
print(queue) # deque([2, 3, 4])

queue.appendleft(1)
print(queue) # deque([1, 2, 3, 4])


# HashSet
# Search them in O(1)
# Insert them in O(1)

mySet = set()

mySet.add(1)
mySet.add(2)
print(mySet) # {1, 2}

print(len(mySet)) # 2

print(1 in mySet) # True
print(2 in mySet) # True
print(3 in mySet) # False

mySet.remove(2)
print(2 in mySet) # False

# List to set
print(set([1, 2, 3]))

# Set comprehension
mySet = {i for i in range(5)}

print(mySet)


# HaspMap (aka dict)

myMap = {}
myMap["alice"] = 88
myMap["bob"] = 77

print(myMap) # {'alice': 88, 'bob': 77}
print(len(myMap))  # 2

myMap["alice"] = 81
print(myMap) # {'alice': 81, 'bob': 77}
print(myMap["alice"]) # 81

print("alice" in myMap) # Search in constant time O(1) True
myMap.pop("alice")

print("alice" in myMap) # Search in constant time O(1) False

myMap = {"alice": 90, "bob": 70}
print(myMap)

# Dict comprehension
myMap = {i : 2*i for i in range(3)}
print(myMap) # {0: 0, 1: 2, 2: 4}

# Looping through maps
print()
myMap = {"alice": 90, "bob": 70}

for key in myMap:
    print(f"{key}: {myMap[key]}")
    # alice: 90
    # bob: 70

for val in myMap.values():
    print(val)
    # 90
    # 70

for key, val in myMap.items():
    print(key, val)



