#Problem_1
a = [9,2,10,1,-1,0,0,1]
L = len(a)-1

def bubbleSort(a):
    x = 0
    for i in range (0,L):
        for j in range (0,L):
            if a[j] > a[j+1]:
                x = a[j]
                a[j] = a[j+1]
                a[j+1] = x


def display(a):
    for item in a:
        print(item)



bubbleSort (a)
display(a)

###########################################################
# Problem_2

x = input()
dict_ = {}

for item in x :
    dict_[item]=0

for item in x:
    dict_[item]= dict_[item]+1

for item in dict_.items():
    print(item[0] , ':' ,item[1])

###########################################################
#Problem_3

x = input()
y = len(x)-1
f=0;
for i in range(0,int(len(x)/2)):
    if not(x[i]== x[y-i]):
        f=1
        break

if f == 0:
    print("true")
else:
    print("false")


