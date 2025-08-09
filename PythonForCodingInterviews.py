A = [1, 2, 3, 4]
print(A) # [1, 2, 3, 4]

# Append - Insert element at the end of array - On average: O(1)
A.append(5)
print(A) # [1, 2, 3, 4, 5]

# Pop - Deleting element at end of array - O(1)
A.pop()
print(A) # [1, 2, 3, 4]

# Insert (not at end of array) - O(n)
A.insert(2, 5)
print(A) # [1, 2, 5, 3, 4]

# Modify an element - O(1)
A[0] = 7
print(A) # [7, 2, 5, 3, 4]

# Accessing element given index i - O(1)
print(A[2]) # 5

# Checking if array has an element - O(n) we have to search through a whole array
if 6 in A:
    print(True)

# Checking length - O(1)
print(len(A)) # 5


# Strings
# Append to end of string - O(n)
s = "hello"
b = s + 'z'
print(b) # helloz

# Check if something is in string - O(n)
if 'f' in s:
    print(True)
print('f' in s) # False

# Access positions - O(1)
print(s[2]) # l

# Check the length of string - O(1)
print(len(s)) # 5