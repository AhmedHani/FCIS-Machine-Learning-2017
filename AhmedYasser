x=[9,2,10,1,-1,0,0,1]
for i in range(len(x)-1):
  for j in range(len(x)-1):
      if x[j] > x[j+1]:
          x[j],x[j+1] = x[j+1],x[j]
print(x)


x=input()
y={}
for i in range(len(x)):
    if not x[i] in y :
        y[x[i]]=0
    y[x[i]]+=1
print(y)

x=input()
y=len(x)
y-=1
for i in range(round(len(x)/2)):
    if not x[i] == x[y]:
        print("false")
        break
    y-=1
if y+i == len(x)-2:
    print ("true")
