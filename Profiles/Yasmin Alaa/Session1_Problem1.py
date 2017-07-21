def bubblesort(arr,n):
    x=0
    for i in range(0,n):
        for j in range(0,n-1):
            if arr[j]>arr[j+1]:
                x=arr[j]
                arr[j]=arr[j+1]
                arr[j+1]=x
    return arr

x=[4,3,2,1]
res=bubblesort(x,4)

for i in range(0, len(res)):
    print(res[i])