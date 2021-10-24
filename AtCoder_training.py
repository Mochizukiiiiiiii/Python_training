#%%
#CODE FESTIVAL 2016 qual B
N, A, B = map(int, input().split())
S = input()
X = A+B
Y = B

for i in range(N):
    if S[i] == 'a':
        if X > 0:
            X -= 1
            print('Yes')
        else:
            print('No')
    
    elif S[i] == 'b':
        if X > 0 and Y > 0:
            X -= 1
            Y -= 1
            print('Yes')
        else:
            print('No')
    
    else:
        print('No')
# %%
