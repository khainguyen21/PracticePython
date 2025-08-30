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
from typing import Optional

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
from scipy.cluster.hierarchy import correspond

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

# Arrays (called lists in python)
arr = [1, 2, 3]
print(arr) # [1, 2, 3]

# Can be used as a stack
arr.append(4)
arr.append(5)
print(arr) # [1, 2, 3, 4, 5]

arr.pop()
print(arr) # [1, 2, 3, 4]

print()
# Technically this is an array not a stack we can insert into the middle O(n)
arr.insert(1, 7)
print(arr) # [1, 7, 2, 3, 4]

arr[0] = 0 # O(1)
arr[3] = 0 # O(1)
print(arr[0]) # O(1) 0
print(arr) # O(1) [0, 7, 2, 0, 4]

print()
# Initialize arr of size n with default value of 1
n = 5
arr = [1] * n
print(arr) # [1, 1, 1, 1, 1]
print(len(arr)) # 5

print()
# Careful: -1 is not out of bounds
# it's the last value
arr = [1, 2, 3]
print(arr[-1]) # 3
# Indexing -2 is the second to last value, etc
print(arr[-2]) # 2


# Sublists (aka slicing)
arr = [1, 2, 3, 4]
print(arr[1:3]) # from index 1 to index 3 not including 3 just like for loop [2, 3]

# Similar to for-loop ranges, last index is non-inclusive
print(arr[0:4]) # [1, 2, 3, 4]


# Unpacking
a, b, c = [1, 2, 3]
print(a, b, c) # 1 2 3

# Be careful
#  a, b = [1, 2, 3] # The amount of left side has to be matched along with the right side

print()
# Loop through arrays
nums = [1, 2, 3]

# Using index
for i in range(len(nums)):
    print(nums[i])

# Without index
for num in nums:
    print(num)

# With index and value
for i, n in enumerate(nums):
    print(f"at index {i}, value is: {n}")
    # at index 0, value is: 1
    # at index 1, value is: 2
    # at index 2, value is: 3
# Loop through multiple arrays simultaneously
# with unpacking
nums1 = [1, 3, 5]
nums2 = [2, 4, 6]

for n1, n2 in zip(nums1, nums2):
    print(n1, n2)
    # 1 2
    # 3 4
    # 5 6
# Reverse
nums = [1, 2, 3]
nums.reverse()
print(nums) # [3, 2, 1]

# Sorting
arr = [5, 4, 7, 3, 8]
arr.sort()
print(arr) # [3, 4, 5, 7, 8]

arr.sort(reverse=True)
print(arr) # [8, 7, 5, 4, 3]

arr = ["bob", "alice", "jane", "doe"]
arr.sort()
print(arr) #['alice', 'bob', 'doe', 'jane']


# Customer sort (by the length of string)
arr.sort(key=lambda x: len(x))
print(arr) #['bob', 'doe', 'jane', 'alice']


# List comprehesion
arr =  [i for i in range(5)]
print(arr) # [0, 1, 2, 3, 4]
arr =  [i+1 for i in range(5)]
print(arr) # [1, 2, 3, 4, 5]


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
#set (HashSet): A set stores only unique elements.
# It's an unordered collection with no duplicate items.
# You can't have the same element in a set more than once.

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
# dict (HashMap): A dictionary stores data as pairs.
# Each item consists of a unique key and its corresponding value.
# You use the key to look up the value.

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


# Tuples are like arrays but immutable
tup = (1, 2, 3)
print(tup)
print(tup[0])
print(tup[-1])

# Cannot modify
#tup[0] = 0 # This cannot work

# You'll mainly be using tuples as keys a hashmap or hashset
myMap = {(1,2): 3}
print(myMap[(1, 2)])
print(myMap)

print()
# Heaps
import heapq

# under the hood are arrays
minHeap = []

# default the heap in python is minheap
heapq.heappush(minHeap, 3)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 4)

# minimum value is always at index 0
print(minHeap[0])
while len(minHeap):
    print(heapq.heappop(minHeap))


print()
# NO max heaps by default, work around is
# to use min heap and multiply by -1 when
# push & pop
maxHeap = []
heapq.heappush(maxHeap, -3)
heapq.heappush(maxHeap, -2)
heapq.heappush(maxHeap, -4)

# Max is always at index 0
print(-1 * maxHeap[0])

while len(maxHeap):
    print(-1 * heapq.heappop(maxHeap))

print()
# Build head from initial values
arr = [2, 1, 8, 4, 5]
heapq.heapify(arr) # O(n)

while arr: # while the array is not empty
    print(heapq.heappop(arr))


# Functions
def myFunc(n, m):
    return n * m

# Nested functions have access to outer
# variables
def outer (a, b):
    c = "c"

    def inner():
        return a + b + c # Still have access to c variable
    return inner()

print(outer("a", "b"))

print()
def romanToInt(s):
    map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    result = 0
    i = 0
    while i < len(s):
        if i < len(s) - 1 and map[s[i]] < map[s[i + 1]]:
            result += map[s[i + 1]] - map[s[i]]
            i += 1
        else:
            result += map[s[i]]
        i += 1

    return result


print(romanToInt("XXIVI"))


def longestCommonPrefix(strs):
    res = ""
    firstStr = strs[0]
    for i in range(len(firstStr)):
        for s in strs:
            if s[i] != firstStr[i]:
                return res

        res += (firstStr[i])


print(longestCommonPrefix(["bat","bag","bank","band"]))

print()

def removeElement(nums, val):

    for num in nums:
        if val == num:
            nums.remove(num)

    return len(nums)

print(removeElement([1,1,2,3,4],1))


def isPalindrome(s):
    left = 0
    right = len(s) - 1

    while (left < right and isAlphaNum(s[right]) and isAlphaNum(s[left])):
        if s[right].lower() != s[left].lower():
            return False

        left += 1
        right -= 1

    return True


def isAlphaNum(c):
    return (ord('0') <= ord(c) <= ord('1') or
            ord('A') <= ord(c) <= ord('Z') or
            ord('a') <= ord(c) <= ord('z'))


isPalindrome("race a car")


def majorityElement( nums):
    #thresHold = len(nums) / 2
    myMap = {}

    res, maxCount = 0, 0
    for i, num in enumerate(nums):
        if num not in myMap:
            myMap[num] = 1

        else:
            myMap[num] += 1

        if myMap[num] > maxCount:
            res = num

        maxCount = max(myMap[num], maxCount)

    return res


majorityElement([5,5,1,1,1,5,5])


def isValid(s):
    stack = []

    if len(s) % 2 != 0:
        return False

    for c in s:
        if s[i] == '{' or s[i] == '(' or s[i] == '[':
            stack.append(s[i])


        elif not stack:
            return False
        else:
            if s[i] == '}' and '{' != stack.pop():
                return False
            elif s[i] == ')' and '(' != stack.pop():
                return False
            elif s[i] == ']' and '[' != stack.pop():
                return False
            else:
                return False
    return len(stack) == 0

print(isValid("([{}])"))


def plusOne(digits) :
    digits = digits[::-1]
    one, i = True, 0
    while one:
        if i < len(digits):
            if digits[i] == 9:
                digits[i] = 0
                i += 1
            else:
                digits[i] += 1
                one = False
        else:
            digits.append(1)


    return digits[::-1]


print(plusOne([9]))

print()

def intersect(nums1, nums2) :
    p1, p2 = 0, 0
    nums1 = sorted(nums1)
    nums2 = sorted(nums2)
    res = []
    while p1 < len(nums1) and p2 < len(nums2):
        if nums1[p1] == nums2[p2]:
            res.append(nums1[p1])
            p1 += 1
            p2 += 1

        elif nums1[p1] > nums2[p2]:
            p2 += 1
        else:
            p1 += 1

    return res
print(intersect([4,9,5], [9,4,9,8,4]))

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def hasCycle(head: Optional[ListNode]) -> bool:
    # Floyd's Tortoise and Hare algorithm
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

def list_to_linked(values):
    # Helper to convert a Python list into a linked list and return the head
    dummy = ListNode(0)
    curr = dummy
    for v in values:
        curr.next = ListNode(v)
        curr = curr.next
    return dummy.next

def isPalindrome(head: Optional[ListNode]):
    slow , fast = head , head
    stack = []
    curr = head
    while fast:

        if slow.val != fast.val:
            stack.append(slow.val)
            slow = curr.next
            fast = fast.next
        else:
            while fast:
                if not stack or fast.next.val != stack.pop():
                    return False
                fast = fast.next

    return True


def deleteDuplicates(head: Optional[ListNode]) :
    prev = head
    curr = head.next

    while curr:
        if prev.next.val != curr.val:
            prev.next = curr
            prev = curr
        curr = curr.next

    return prev
#
# if __name__ == "__main__":
#     # Build a linked list and pass the head node
#     head1 = list_to_linked([1, 2])
#     print(isPalindrome(head1))  # True
#
#     print(deleteDuplicates(list_to_linked[1,1,2,3,3]))


def countStudents(students, sandwiches):
    student_queue = deque(students)
    total_tries = len(students)
    s_idx = 0
    while total_tries > 0 and s_idx < len(sandwiches):

        if student_queue[0] != sandwiches[s_idx]:

            front_student = student_queue.popleft()

            student_queue.append(front_student)

            total_tries -= 1

        else:
            student_queue.popleft()
            s_idx += 1
            total_tries = len(student_queue)

    return len(student_queue)


print(countStudents([1,1,0,0], [0,1,0,1]))


def addDigits(num):
    if num == 0:
        return 0

    while num >= 10:
        res = 0
        while num >= 1:
            res += num % 10
            num //= 10
        num = res

    return num

print(addDigits(38))


def isHappy(n: int):
    count = set()

    while n not in count:
        count.add(n)
        res = 0

        while n:
            res += (n % 10) ** 2
            n //= 10

        n = res

        if res == 1:
            return True

    return False


print(isHappy(19))