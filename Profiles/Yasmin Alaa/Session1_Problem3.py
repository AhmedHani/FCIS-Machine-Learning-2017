x="aba"
j=len(x)-1
for i in range(0,len(x)):
       if j<=i:
              print("yes")
              break
       elif x[i]!= x[j]:
              print("no")
              break
       else:
              j-=1