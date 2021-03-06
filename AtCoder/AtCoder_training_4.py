# %%
#ABC235 A
abc = input()
a = int(abc)
b = int(abc[1]+abc[2]+abc[0])
c = int(abc[2]+abc[0]+abc[1])
print(a+b+c)
# %%
#ABC235 B
from collections import deque
from curses import mouseinterval

N = int(input())
H = list(map(int, input().split()))
Q = deque(H)

q = Q.popleft()

for _ in range(N-1):
    if q < Q[0]:
        q = Q.popleft()
    else:
        print(q)
        exit()

print(q)
# %%
#ABC235 C
from collections import defaultdict
N, Q = map(int, input().split())
A = list(map(int, input().split()))
m = defaultdict(list)
for i in range(N):
  m[A[i]].append(i + 1)
for _ in range(Q):
  x, k = map(int, input().split())
  if k <= len(m[x]):
    print(m[x][k - 1])
  else:
    print(-1)
# %%
#ABC235 D
from collections import defaultdict, deque

a, N = map(int, input().split())

cnt = 0
d = defaultdict(lambda:-1)
d[1] = 0
Q = deque()
Q.append([1, 0])

while Q:
    q, cnt = Q.popleft()
    cnt += 1

    q_new = q * a
    if len(str(q_new)) <= len(str(N)) and (d[q_new] > cnt or d[q_new] == -1):
        d[q_new] = cnt
        Q.append([q_new, cnt])
    
    if q >= 10 and q%10 != 0:
        q_new = int(str(q)[-1] + str(q)[:-1])
        if d[q_new] > cnt or d[q_new] == -1:
            d[q_new] = cnt
            Q.append([q_new, cnt])

print(d[N])
# %%
#DPL_1_B
#ナップザックDP
N, W = map(int, input().split())
v = []
w = []
for _ in range(N):
    x,y = map(int,input().split())
    v.append(x)
    w.append(y)

dp = [[0]*(W+1) for j in range(N+1)]

for i in range(N):
    for j in range(W+1):
        if j < w[i]:
            dp[i+1][j] = dp[i][j]
        else:
            dp[i+1][j] = max(dp[i][j],dp[i][j-w[i]]+v[i])

print(dp[N][W])
# %%
#DPL_1_C
#ナップザックDP
N, W = map(int, input().split())
v = []
w = []
for _ in range(N):
    x,y = map(int,input().split())
    v.append(x)
    w.append(y)

dp = [[0]*(W+1) for _ in range(N+1)]

for i in range(N):
    for j in range(W+1):
        if j < w[i]:
            dp[i+1][j] = dp[i][j]
        else:
            dp[i+1][j] = max(dp[i][j],dp[i+1][j-w[i]]+v[i])

print(dp[N][W])
# %%
#DPL_1_A
#ナップザックDP
n, m = map(int, input().split())
c = list(map(int, input().split()))

dp = [[float('inf')] * (n+1) for _ in range(m)]
for i in range(m):
    dp[i][0] = 0
for j in range(1 + n):
    dp[0][j] = j

for i in range(m-1):
    for j in range(n+1):
        dp[i+1][j] = min(dp[i+1][j], dp[i][j])
        if 0 <= j - c[i+1]:
            dp[i+1][j] = min(dp[i+1][j], dp[i+1][j-c[i+1]] + 1)

print(dp[m-1][n])
# %%
#ALDS1_10_C
#ナップザックDP
q = int(input())

for _ in range(q):
  X = input()
  Y = input()
  dp = [[0] * (len(X)+1) for _ in range(len(Y)+1)]

  for i in range(len(Y)):
    for j in range(len(X)):
      if X[j] == Y[i]:
        dp[i+1][j+1] = dp[i][j] + 1
      else:
        dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
  
  print(dp[-1][-1])
# %%
#第１０回日本情報オリンピック 予選（過去問） D
#ナップザックDP
N = int(input())
A = list(map(int, input().split()))

dp = [[0] * 21 for _ in range(N-1)]
dp[0][A[0]] = 1
A = A[1:]

for n in range(N-2):
  for m in range(21):
    if m - A[n] >= 0:
      dp[n+1][m] += dp[n][m-A[n]]
    if m+A[n] <= 20:
      dp[n+1][m] += dp[n][m+A[n]]

print(dp[-1][A[-1]])
# %%
#第１１回日本情報オリンピック 予選（過去問） D
#ナップザックDP
MOD = 10**4
N, K = map(int, input().split())
A = [0] * N

for _ in range(K):
  a, b = map(int, input().split())
  A[a-1] = b
dp = [[[0]*4 for _ in range(4)] for _ in range(N+1)]
dp[0][0][0] = 1
for n in range(N):
  for i in range(4):
    for j in range(4):
      for k in range(1, 4):
        if A[n] != 0 and A[n] != k:
          continue
        if k != i or i != j:
          dp[n+1][k][i] += dp[n][i][j]
          dp[n+1][k][i] %= MOD

ans = 0
for i in range(4):
  for j in range(4):
    ans += dp[-1][i][j]
    ans %= MOD

print(ans)
# %%
#ABC204 C
#BFS
from collections import deque

N, M = map(int, input().split())
G = [[] for _ in range(N+1)]

for _ in range(M):
  A, B = map(int, input().split())
  G[A].append(B)

def bfs(s):
  Q = deque()
  visited = [False] * (N+1)
  visited[s] = True
  Q.append(s)

  while Q:
    now = Q.popleft()
    for to in G[now]:
      if visited[to] == False:
        visited[to] = True
        Q.append(to)
  
  return sum(visited)

ans = 0
for i in range(1, N+1):
  ans += bfs(i)

print(ans)
# %%
#ABC204 D
#部分和 DP
N = int(input())
T = [0] + list(map(int, input().split()))

MAX = sum(T)

dp = [[False] * (MAX+1) for _ in range(N+1)]
dp[0][0] = True

for i in range(1, N+1):
  for j in range(MAX+1):
    if dp[i-1][j] == True:
      dp[i][j] = True
    
    if 0 <= j-T[i] and dp[i-1][j-T[i]] == True:
      dp[i][j] = True

ans = 10**20

for i in range(MAX+1):
  if dp[N][i] == True:
    ans = min(ans, max(i, MAX-i))
  
print(ans)
# %%
#ABC203 C
N, K = map(int, input().split())

AB = [tuple(map(int, input().split())) for _ in range(N)]
AB.sort()

for i in range(N):
  A = AB[i][0]
  B = AB[i][1]

  if A <= K:
    K += B
  else:
    break

print(K)
# %%
#ABC233 E
X = input()
N = len(X)

X_list = list(X)

for i in range(N):
  X_list[i] = int(X_list[i])

d = [0]*N
d[0] = X_list[0]

for i in range(1, N):
  d[i] = d[i-1] + X_list[i]

for i in range(N-1, 0, -1):
  d[i-1] += d[i]//10
  d[i]=str(d[i])
  d[i]=d[i][-1]

d[0] = str(d[0])

ans = ''.join(d)

print(ans)
# %%
#ABC218 E
#UnionFind

class UnionFind:
    def __init__(self,n):
        self.n=n
        self.parent_size=[-1]*n

    def leader(self,a):
        if self.parent_size[a]<0: return a
        self.parent_size[a]=self.leader(self.parent_size[a])
        return self.parent_size[a]

    def merge(self,a,b):
        x,y=self.leader(a),self.leader(b)
        if x == y: return 
        if abs(self.parent_size[x])<abs(self.parent_size[y]):x,y=y,x
        self.parent_size[x] += self.parent_size[y]
        self.parent_size[y]=x
        return 

    def same(self,a,b):
        return self.leader(a) == self.leader(b)

    def size(self,a):
        return abs(self.parent_size[self.leader(a)])

    def groups(self):
        result=[[] for _ in range(self.n)]
        for i in range(self.n):
            result[self.leader(i)].append(i)
        return [r for r in result if r != []]

N, M = map(int, input().split())
G = []
ans = 0

for _ in range(M):
  A, B, C = map(int, input().split())
  G.append([C, A, B])

  if 0 < C:
    ans += C

G.sort()

UF = UnionFind(N+1)

for c, a, b in G:
  if c <= 0:
    UF.merge(a, b)
  
  else:
    if UF.same(a, b) == False:
      ans -= c
      UF.merge(a, b)

print(ans)
# %%
#ABC236 C
from collections import deque

N, M = map(int, input().split())
S = list(map(str, input().split()))
T = list(map(str, input().split()))

Sq = deque(S)
Tq = deque(T)

for _ in range(N):
  s = Sq.popleft()

  if Tq[0] == s:
    print('Yes')
    Tq.popleft()
  
  else:
    print('No')
# %%
#ABC236 D
#DFS 再帰関数
import sys
sys.setrecursionlimit(10**6)

N = int(input())

A = [[0]*(2*N+1) for _ in range(2*N+1)]

for i in range(1, 2*N):
  tmp = list(map(int,input().split()))
  for j in range(len(tmp)):
    A[i][j+(i+1)] = tmp[j]
    A[j+(i+1)][i] = tmp[j]

ans = 0

def DFS(selected, pairs):
  global ans

  if len(pairs) == 2*N:
    score = 0
    for i in range(0, 2*N, 2):
      x=pairs[i]
      y=pairs[i+1]
      score ^= A[x][y]
    
    ans = max(ans, score)

  elif len(pairs)%2 == 0:
    i = 1
    while selected[i] == True:
      i += 1
    
    pairs.append(i)
    selected[i] = True

    DFS(selected, pairs)
    pairs.pop()
    selected[i] = False
  
  else:
    for i in range(1, 2*N+1):
      if selected[i] == False:
        pairs.append(i)
        selected[i] = True
        DFS(selected, pairs)
        pairs.pop()
        selected[i] = False

selected=[False]*(2*N+1)

pairs=[]

DFS(selected, pairs)

print(ans)
# %%
#ABC202 C
from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

A_cnt = defaultdict(int)
for key in A:
    A_cnt[key] += 1

B_cnt = defaultdict(int)

for i in range(N):
  B_cnt[B[C[i]-1]] += 1

ans = 0
for i in range(1, N+1):
  ans += A_cnt[i] * B_cnt[i]

print(ans)
# %%
#ABC202 D
#
#メモ:再帰関数の練習してもいいかも
from math import factorial

A, B, K = map(int, input().split())
ans = []

for i in range(A+B):
  S = factorial(A+B)//factorial(A)//factorial(B)
  if K <= S*A//(A+B):
    ans.append('a')
    A -= 1
  
  else:
    ans.append('b')
    K -= S*A//(A+B)
    B -= 1
    
print(''.join(ans))
# %%
#ABC198 D
#TLE
import copy
from itertools import permutations

S1 = list(input())
S2 = list(input())
S3 = list(input())

num1 = copy.copy(S1)
num2 = copy.copy(S2)
num3 = copy.copy(S3)

chr = set()
chr.update(S1)
chr.update(S2)
chr.update(S3)

if len(chr) > 10:
  print('UNSOLVABLE')
  exit()

keys = list(chr)
values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
values = list(permutations(values, len(keys)))

for value in values:
  mydict = dict(zip(keys, value))

  for i in range(len(S1)):
    num1[i] = mydict[S1[i]]
  
  for i in range(len(S2)):
    num2[i] = mydict[S2[i]]
  
  for i in range(len(S3)):
    num3[i] = mydict[S3[i]]
  
  if num1[0] == '0' or num2[0] == '0' or num3[0] == '0':
    continue

  num1_ = int(''.join(num1))
  num2_ = int(''.join(num2))
  num3_ = int(''.join(num3))

  if num1_ + num2_ == num3_:
    print(num1_)
    print(num2_)
    print(num3_)
    exit()

print('UNSOLVABLE')

# %%
#ABC197 B
#BFS
from collections import deque

H, W, X, Y = map(int, input().split())
S = [[] for _ in range(H+2)]
S[0] = ['#'] * (W+2)
S[-1] = ['#'] * (W+2)

for i in range(1, H+1):
  S[i] = ['#'] + list(input()) + ['#']

visited = [[False] * (W+2) for _ in range(H+2)]

Q = deque()
Q.append([X, Y])
visited[X][Y] = True
ans = 1

while Q:
  X_, Y_ = Q.popleft()
  for x, y in [(1, 0), (-1, 0)]:
    toX = X_+x
    toY = Y_+y
    if S[toX][toY] == '.' and visited[toX][toY] == False:
      visited[toX][toY] = True
      Q.append([toX, toY])
      ans += 1

visited = [[False] * (W+2) for _ in range(H+2)]
Q.append([X, Y])
visited[X][Y] = True

while Q:
  X_, Y_ = Q.popleft()
  for x, y in [(0, 1), (0, -1)]:
    toX = X_+x
    toY = Y_+y
    if S[toX][toY] == '.' and visited[toX][toY] == False:
      visited[toX][toY] = True
      Q.append([toX, toY])
      ans += 1

print(ans)
# %%
#ABC197 C
#bit全探索
N = int(input())
A = list(map(int, input().split()))

ans = float('INF')

if len(A) == 1:
  print(A[0])
  exit()

for bit in range(2**(N-1)):
  log_sum = A[0]
  xor = 0

  for i in range(1, N):
    if (bit >> (i-1)) & 1:
      xor ^= log_sum
      log_sum = 0
      log_sum |= A[i]
    else:
      log_sum |= A[i]
  
  xor ^= log_sum
  ans = min(ans, xor)

print(ans)
# %%
#ABC197 D
import numpy as np

N = int(input())
x0, y0 = map(int, input().split())
x, y = map(int, input().split())

Px, Py = (x0+x)/2, (y0+y)/2
vec = ((x0-x)/2, (y0-y)/2)

def rotation_o(u, t, deg=True):

    # 度数単位の角度をラジアンに変換
    if deg == True:
        t = np.deg2rad(t)

    # 回転行列
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])

    return  np.dot(R, u)

R_vec = rotation_o(vec, 360/N)

ans = [Px + R_vec[0], Py + R_vec[1]]

print(*ans)
# %%
#ABC197 E
import sys

N = int(input())
ball = [[] for _ in range(N+1)]
for _ in range(N):
  X, C = map(int, sys.stdin.readline().split())
  C = -1
  ball[C].append(X)
ball[N].append(0)
for i in range(N+1):
  ball[i].sort()

INF = float('inf')

dp = [[INF] * 2 for _ in range(N+2)]
dp[0][0] = dp[0][1] = 0

pos = [[None] * 2 for _ in range(N+2)]
pos[0][0] = pos[0][1] = 0

L, R = 0, -1
for i in range(N+1):
  if ball[i]:
    for j in [L, R]:
      crt_cost = dp[i][j]
      crt_pos = pos[i][j]


# %%
N = int(input())
K = len(str(N))

ans = 0

for i in range(K-1):
  ans += (1+9*(10**i)) * 9*(10**i)//2

s = N - 10 ** (K-1) + 1

ans += (1+s)*s//2

print(ans%998244353)
# %%
from collections import deque

T = int(input())
for _ in range(T):
  A, S = map(int,input().split())
  
  if A > S:
    print('No')
    continue

  bin_A = format(A, 'b')
  bin_A = bin_A.zfill(60)
  bin_S = format(S, 'b')
  bin_S = bin_S.zfill(60)

  Q_A = deque(list(bin_A))
  Q_S = deque(list(bin_S))
  c = 0
  flag = 1

  for _ in range(60):
    a, s = int(Q_A.pop()), int(Q_S.pop())

    if c == 0 and a == 1 and s == 1:
      flag = 0
    if c == 1 and a == 1 and s == 0:
      flag = 0

    if (c == 1 and ((a == 0 and s == 0) or (a == 1 and s == 1))) or (c == 0 and a == 1 and s == 0):
      c = 1
    else:
      c = 0
  
  if flag:
    print('Yes')
  
  else:
    print('No')
# %%
#ITP_7_B
#全探索
flag = 1

while flag:
  n, x = map(int,input().split())
  if n == 0 and x == 0:
    flag = 0
  else:
    ans = 0
    for i in range(1, n+1):
      for j in range(i+1, n+1):
        for k in range(j+1, n+1):
          if i+j+k == x:
            ans += 1
    print(ans)
# %%
#ABC106 B
#全探索
N = int(input())
ans = 0

for i in range(1, N+1, 2):
  cnt = 0
  for j in range(1, i+1, 2):
    if i%j == 0:
      cnt += 1
  if cnt == 8:
    ans += 1

print(ans)
# %%
#ABC122 B
#全探索
S = input()
ans = 0
std = 0
ans = 0
for l in S:
  if l in 'ACGT':
    std += 1
  else:
    ans = max(ans,std)
    std = 0
ans = max(ans,std)

print(ans)
# %%
#パ研杯2019 C
#全探索
N, M = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(N)]
ans = 0

for i in range(M):
  for j in range(i+1, M):
    cnt = 0
    for k in range(N):
      cnt += max(A[k][i], A[k][j])
    ans = max(ans, cnt)

print(ans)
# %%
#ABC095 C
#全探索
A, B, C, X, Y = map(int, input().split())

N = max(X, Y)
ans = float('inf')

for i in range(N+1):
  x = max(0, X-i)
  y = max(0, Y-i)
  
  ans = min(ans, A*x+B*y+2*C*i)
  
print(ans)
# %%
#三井住友信託銀行プログラミングコンテスト2019 D
#全探索
N = int(input())
S = input()
ans = 0

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
      ans += 1

print(ans)
# %%
#ALDS1_5_A
#ビット全探索
n = int(input())
A = list(map(int, input().split()))
q = int(input())
mi = list(map(int, input().split()))

myset = set()

for bit in range(1 << n):
  cnt = 0
  for i in range(n):
    if bit & (1 << i):
      cnt += A[i]
  myset.add(cnt)

for m in mi:
  if m in myset:
    print('yes')
  else:
    print('no')
# %%
#ABC239 C
from collections import deque

x1, y1, x2, y2 = map(int, input().split())
set1 = set()
set2 = set()

Q1 = deque()
Q1.append([x1, y1])
x, y = Q1.popleft()
for dx, dy in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
  X, Y = x+dx, y+dy
  Q1.append([X, Y])
  set1.add(tuple([X, Y]))

Q2 = deque()
Q2.append([x2, y2])
x, y = Q2.popleft()
for dx, dy in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
  X, Y = x+dx, y+dy
  Q2.append([X, Y])
  set2.add(tuple([X, Y]))

if set1.isdisjoint(set2):
  print('No')
else:
  print('Yes')
# %%
#ABC239 D
#エラトステネスの篩
A, B, C, D = map(int, input().split())

prime = [True]*201
prime[0] = [False]
prime[1] = [False]
for p in range(15):
  if prime[p]:
    for i in range(p*p, 201, p):
      prime[i] = False

for i in range(A, B+1):
  if all(not prime[i+j] for j in range(C, D+1)):
    print('Takahashi')
    exit()

print('Aoki')
# %%
#ABC239 E
#DFS 再帰関数
import sys

readline = sys.stdin.readline
sys.setrecursionlimit(10 ** 6)

MAX_K = 20

def dfs(u, p):
  L[u].append(X[u])
  for v in G[u]:
    if v == p:
      continue
    dfs(v, u)
    L[u].extend(L[v])
  L[u].sort(reverse=True)
  L[u] = L[u][:MAX_K]

N, Q = map(int, input().split())
X = [0] + list(map(int, input().split()))
G = [[] for _ in range(N+1)]
for _ in range(N-1):
  a, b = map(int, input().split())
  G[a].append(b)
  G[b].append(a)
L = [[] for _ in range(N+1)]
dfs(1, 0)

for _ in range(Q):
  v, k = map(int, input().split())
  print(L[v][k-1])
# %%
#ABC240 C
#DP
N, X = map(int, input().split())
dp = [[False]*10001 for _ in range(N+1)]
dp[0][0] = True

for i in range(1, N+1):
  a, b = map(int, input().split())
  for j in range(0, 10001):
    if dp[i-1][j]:
      dp[i][j+a] = True
      dp[i][j+b] = True

if dp[-1][X]:
  print('Yes')
else:
  print('No')
# %%
#ABC240 D
from collections import deque

N = int(input())
a = deque(map(int, input().split()))
ans = 0
num_Q = deque()

for _ in range(N):
  num = a.popleft()
  if ans == 0:
    num_Q.append(num)
    cnt_Q = deque([1])
    ans = 1
    print(ans)
  
  elif num != num_Q[-1]:
    num_Q.append(num)
    cnt_Q.append(1)
    ans += 1
    print(ans)
  else:
    if cnt_Q[-1] == num - 1:
      num_Q.pop()
      cnt_Q.pop()
      ans -= (num - 1)
      print(ans)
    else:
      tmp = cnt_Q.pop()
      cnt_Q.append(tmp+1)
      ans += 1
      print(ans)
      
# %%
#ABC241 C
#全探索
N = int(input())
S = [[1 if c == "#" else 0 for c in input()] for _ in range(N)]
pat = [(1, 0), (0, 1), (1, -1), (1, 1)]

def judge(sy, sx, dy, dx):
  y, x = sy, sx
  cnt = 0
  for _ in range(6):
    if not (0 <= y < N and 0 <= x < N):
      return False
    cnt += S[y][x]
    y += dy
    x += dx
  return cnt >= 4

for y in range(N):
  for x in range(N):
    for dy, dx in pat:
      if judge(y, x, dy, dx):
        print('Yes')
        exit()

print('No')
# %%
#ABC242 C
#DP

MOD = 998244353
N = int(input())

dp = [[0] * 11 for _ in range(N + 1)]
for i in range(1, 10):
  dp[1][i] = 1

for i in range(1, N):
  for j in range(1, 10):
    dp[i+1][j] = (dp[i][j - 1] + dp[i][j] + dp[i][j + 1]) % MOD

print(sum(dp[N])%MOD)
# %%
#ABC243 C
from collections import defaultdict
INF = float('inf')
flag = 0

N = int(input())
XY = [list(map(int, input().split())) for _ in range(N)]
S = input()
L_max = defaultdict(lambda: -INF)
R_min = defaultdict(lambda: INF)

for s, (x, y) in zip(S, XY):
  if s == 'L':
    L_max[y] = max(L_max[y], x)
  else:
    R_min[y] = min(R_min[y], x)
    
for y in L_max.keys():
  if R_min[y] < L_max[y]:
    flag = 1

if flag:
  print('Yes')
else:
  print('No')
# %%
#ABC243 D
N, X = map(int, input().split())
S = input()
T = ['!']

for s in S:
  if s == 'U' and T[-1] in 'LR':
    T.pop()
  else:
    T.append(s)

ans = X
for t in T[1:]:
  if t == 'U':
    ans //= 2
  elif t == 'L':
    ans *= 2
  else:
    ans *= 2
    ans += 1

print(ans)
# %%
#ABC243 D
N, X = map(int, input().split())
S = input()
V = [*f'{X:b}']
#V = [c for in bin(X)[2:]]
#V = [*format(X, 'b')]

for s in S:
  if s == 'U':
    V.pop()
  elif s == 'L':
    V += ['0']
  else:
    V += ['1']

concat = lambda L: ''.join(L)
Vs = concat(V)

print(Vs)
# %%
#ABC244 C
N = int(input())
nums = set(range(1, 2*N+2))
flag = 1

while flag:
  print(nums.pop(), flush=True)
  A = int(input())
  if A != 0:
    nums.remove(A)
  else:
    flag = 0
# %%
#ABC244 D
S1, S2, S3 = map(str, input().split())
T1, T2, T3 = map(str, input().split())

if (T1, T2, T3) in ((S1, S2, S3), (S2, S3, S1), (S3, S1, S2)):
  print('Yes')
else:
  print('No') 
# %%
# test