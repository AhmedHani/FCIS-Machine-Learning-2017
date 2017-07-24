#problem 1
z=raw_input()
x=map(int,z.split())
for i in range(len(x)-1):
  for j in range(len(x)-1):
      if x[j] > x[j+1]:
          x[j],x[j+1] = x[j+1],x[j]
print(x)

-------------------------------------------------------------------------------------------------------------------------------

#problem 2
from collections import Counter
from numpy import *
x=raw_input()
print (Counter(x))

-------------------------------------------------------------------------------------------------------------------------------

#problem 3
s=raw_input()
ss=""
for i in range(len(s)-1,-1,-1):
    ss+=s[i]
if s==ss:
    print ("YES")
else :
    print ("NO")
