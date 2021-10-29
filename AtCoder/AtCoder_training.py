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
#ABC074 B
N = int(input())
K = int(input())
x = list(map(int, input().split()))
ans = 0

for i in range(N):
    ans += 2*min(abs(x[i]), abs(x[i]-K))

print(ans)
# %%
#ABC068 B
N = int(input())
ans = 1
max_count = 0
for i in range(1, N+1):
    count = 0
    tmp = i
    while tmp % 2 == 0:
        tmp /= 2
        count += 1
    if max_count < count:
        max_count = count
        ans = i

print(ans)
# %%
#ABC160 C
K, N = map(int, input().split())
A = list(map(int, input().split()))

distance = A[0]+K-A[N-1]
for i in range(N-1):
    distance = max(distance, A[i+1]-A[i])

print(K - distance)
# %%
#パ研杯2019 C
N, M = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(N)]
ans = 0
score = 0

for i in range(M):
    for j in range(i+1, M):
        for k in range(N):
            score += max(A[k][i], A[k][j])
            
        ans = max(ans, score)
        score = 0
print(ans)
# %%
#三井住友信託銀行プログラミングコンテスト2019 D
N = int(input())
S = input()
count = 0

for i in range(10):
    for j in range(10):
        for k in range(10):
            pin = str(i) + str(j) + str(k)
            if S.find(pin[0]) == -1:
                continue
            S1 = S[S.find(pin[0])+1:]
            if S1.find(pin[1]) == -1:
                continue
            S2 = S1[S1.find(pin[1])+1:]
            if S2.find(pin[2]) == -1:
                continue
            count += 1

print(count)
# %%
#ALDS_5_A
n = int(input())
A = list(map(int, input().split()))
q = int(input())
m = list(map(int, input().split()))
hash_table = {}

for i in range(1 << n):
    tmp_cnt = 0
    for j in range(n):
        if (i >> j & 1):
            tmp_cnt += A[j]
    hash_table[tmp_cnt] = 1

for i in m:
    if hash_table.get(i):
        print('yes')
    else:
        print('no')
# %%
#bit全探索参考
S = input() #l桁の数字を文字列型で入力
l = len(S) # 文字列の長さ
n = l - 1 # 文字列の間の数
ans = 0
for bit in range(1 <<  n): # 0から((1をnだけ右シフトした数)-1)までのfor文
    s = S[0]
    for i in range(n):
        if (bit & (1 << i)): # 0から(2^n - 1)までのfor文
            s += '+'
        s += S[i +1]
    ans += sum(map(int, s.split('+')))
print(ans)