
"""question 1  bubble sort"""
def bubble_sort(x,n):

    for i in range(0,n-1):

        for y in range(0,n-1):
            if x[y] > x[y+1]:
                temp = x[y]
                x[y] = x[y+1]
                x[y+1] = temp

    print(x)
    return x

bubble_sort([9,2,10,1,-1,0,0,1],8)


"""question 2  get characters' frequencies in a string"""
def get_frequency(S):
    freq = dict()
    for i in S:
        if i in freq :
            freq[i]+=1
        else:
            freq[i] = 1
    print(freq)
    return  freq

get_frequency('aaabbbb')


"""question 3  check if a sring is palindrome or not"""

def is_palindrome(S):
    for x,y in zip( range(0,len(S)),range(len(S)-1,0,-1)):
        if S[x] != S[y]:
            print("False")
            return False
            break
    print("True")
    return True

is_palindrome("adddd")



