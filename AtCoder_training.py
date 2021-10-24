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
#三井住友信託銀行プログラミングコンテスト2019 B
import math

N = int(input())
ans = ':('
for i in range(1, N+1):
    if math.floor(i*1.08) == N:
        ans = i
        flag = True

print(ans)
# %%
#ABC121 B
N, M, C = map(int, input().split())
B = list(map(int, input().split()))
A = [list(map(int, input().split())) for i in range(N)]
count = 0

for i in range(N):
    product = [a*b for a,b in zip(A[i], B)]
    calc = sum(product)+C
    if calc > 0:
        count += 1

print(count)

# %%
#パナソニックプログラミングコンテスト2020 B
H, M = map(int, input().split())

if min(H, M) == 1:
    ans = 1

elif H*M%2 == 0:
    ans = H*M//2

else:
    ans = (H*M+1)//2

print(ans)
# %%
#ABC157 B
A = [list(map(int,input().split())) for l in range(3)]
N = int(input())

for i in range(N):
    b = int(input())
    for j in range(3):
        for k in range(3):
            if A[j][k] == b:
                A[j][k] = 0

for i in range(3):
    if A[i][0] == A[i][1] == A[i][2] == 0:
        print('Yes')
        exit()
    elif A[0][i] == A[1][i] == A[2][i] == 0:
        print('Yes')
        exit()

if A[0][0] == A[1][1] == A[2][2] == 0:
    print('Yes')
    exit()
elif A[0][2] == A[1][1] == A[2][0] == 0:
    print('Yes')
    exit()
else:
    print('No')
# %%
#ABC086 B
a, b = map(int, input().split())
ans = 'No'
for i in range(1, 400):
    if i**2 == int(str(a)+str(b)):
        ans = 'Yes'

print(ans)
# %%
