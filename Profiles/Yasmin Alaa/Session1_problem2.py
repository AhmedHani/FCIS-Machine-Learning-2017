x="Hello"
dictionary = {}
for i in range(0,len(x)):
    if x[i] in dictionary:
        dictionary[x[i]]+=1
    else:
        dictionary[x[i]]=1
for item in dictionary.items():
    print(item)