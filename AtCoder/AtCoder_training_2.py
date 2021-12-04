# %%
#ABC022 B
N = int(input())
A = [int(input()) for i in range(N)]
flower_flag = set()
cnt = 0

for i in range(N):
    if A[i] in flower_flag:
        cnt += 1
    else:
        flower_flag.add(A[i])

print(cnt)
# %%
#ABC006 C
N = int(input())
fib = [0] * 1000000
fib[2] = 1
if N < 4:
    print(fib[N-1])
    exit()
for i in range(3,N):
    fib[i] = fib[i-3]%10007+fib[i-2]%10007+fib[i-1]%10007
print(fib[N-1]%10007)
# %%
#ABC146 B
N = int(input())
S = list(input())
T = []
alphabets = [chr(i) for i in range(65, 65+26)]
alphabets *= 2
alphabets_dict = dict(zip(alphabets[0:26], alphabets[N:N+26]))

for i in range(len(S)):
    T.append(alphabets_dict[S[i]])

print(''.join(T)) 
# %%
#ABC088 C
C = [list(map(int, input().split())) for i in range(3)]

if C[0][1]-C[0][0] == C[1][1]-C[1][0] == C[2][1]-C[2][0] and C[0][1]-C[0][2] == C[1][1]-C[1][2] == C[2][1]-C[2][2]:
    print('Yes')
else:
    print('No')
# %%
#ABC087 C
N = int(input())
A1 = list(map(int, input().split()))
A2 = list(map(int, input().split()))
ans = 0

for i in range(N):
    candy_sum = sum(A1[0:i+1]) + sum(A2[i:N])
    ans = max(ans, candy_sum)

print(ans)
# %%
#ABC084 C
N = int(input())
CSF = [list(map(int, input().split())) for i in range(N-1)]

# %%
# ABC149 B
A, B, K = map(int, input().split())
if A <= K:
    ans_T = 0
    K -= A
    ans_A = max(B-K, 0)
else:
    ans_T = A-K
    ans_A = B

print(str(ans_T)+' '+str(ans_A))
# %%
#ABC226 A
from decimal import *
import decimal
X = str(input())

print(Decimal(X).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
# %%
#ABC226 B
N = int(input())
a = [list(map(int, input().split())) for a in range(N)]
sets = set()

for i in range(N):
    aa = ''.join(str(a[i]))
    sets.add(aa)

print(len(sets))
# %%
#ABC226 C
import copy
from functools import lru_cache

N = int(input())
TKA = ([list(map(int, input().split())) for a in range(N)])
flag = 1
new_needs = set()

lenN = len(TKA[N-1])-2
if lenN == 0:
    ans = TKA[N-1][0]
else:
    new_needs = set(TKA[N-1][2:])

new_needs.add(N)

while flag:
    needs = copy.copy(new_needs)

    for i in list(new_needs):
        if len(TKA[i-1]) != 2:
            new_needs |= set(TKA[i-1][2:])
    
    if new_needs == needs:
        flag = 0

ans = 0
for i in needs:
    ans += TKA[i-1][0]

print(ans)
# %%
#ABC007 C
from collections import deque

R, C = map(int, input().split())
sy, sx = map(int, input().split())
gy, gx = map(int, input().split())
sy, sx, gy, gx = sy-1, sx-1, gy-1, gx-1
c = [[c for c in input()] for _ in range(R)]
visited = [[-1]*C for _ in range(R)]

def bfs(sy, sx, gy, gx, c, visited):
    visited[sy][sx] = 0
    Q = deque([])
    Q.append([sx, sy])
    while Q:
        y, x = Q.popleft()

        if[y, x] == [gy, gx]:
            return visited[y][x]
        
        for i, j in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            if c[y+i][x+j] == '.' and visited[y+i][x+j] == -1:
                visited[y+i][x+j] = visited[y][x]+1
                Q.append([y+i,x+j])

print(bfs(sy, sx, gy, gx, c, visited))
# %%
#ABC158 B
N, A, B = map(int, input().split())
print(N//(A+B)*A + min(N%(A+B), A))
# %%
#ABC138 D
N, Q = map(int, input().split())
tree = [[] for _ in range(N)]

for _ in range(N-1):
    a, b = map(int, input().split())
    tree[a-1].append(b-1)
    tree[b-1].append(a-1)

X = [0]*N
for _ in range(Q):
    p, x = map(int, input().split())
    X[p-1] += x

ans = [0]*N

def dfs(u, parent=None):
    """
    u: 子ノード
    parent: 親ノード
    """
    ans[u] = ans[parent] + X[u]
    for v in tree[u]:
        if v != parent:
            dfs(v, u)

dfs(0, 0)
print(' '.join([str(i) for i in ans]))
# %%
#ABC226 C
def solve():
    # スタックを使ったDFSです（dequeを使ってBFSにしてもいいです）
    seen = [False] * (N + 1)
    stack = [N]
    seen[N] = True

    ans = 0

    while stack:
        u = stack.pop()
        ans += T[u]

        for v in G[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return ans


N = int(input())
T = [0] * (N + 1)
G = [[] for _ in range(N + 1)]

for u in range(1, N + 1):
    T[u], k, *A = map(int, input().split())
    for v in A:
        G[u].append(v)
        #G[v].append(u)  # この行は不要ですが、とりあえず入れておきます（uを覚えるのに必要な技vは、uより小さい番号のため）

print(solve())
# %%
#ABC138 D
from collections import deque
N, q = map(int, input().split())
cnt = [0]*(N+1)
G = [[] for _ in range(N+1)]
for _ in range(N-1):
    a, b = map(int, input().split())
    G[a].append(b)
    G[b].append(a)

for _ in range(q):
    v, val = map(int,input().split())
    cnt[v] += val
q = deque()
q.append(1)
visited = [0]*(N+1)
while q:
    v = q.pop()
    visited[v] = 1
    for u in G[v]:
        if visited[u] == 1:
            continue
        cnt[u] += cnt[v]
        q.append(u)

print(*cnt[1:])
# %%
#ABC226 C
from collections import deque

N = int(input())
T = [0] * (N + 1)
G = [[] for _ in range(N + 1)]

for i in range(1, N+1):
    T[i], k, *A = map(int, input().split())
    for j in A:
        G[i].append(j)
        #G[j].append(i)

def dfs():
    visited = [False] * (N+1)
    stack = deque()
    stack.append(N)
    visited[N] = True

    ans = 0

    while stack:
        u = stack.pop()
        ans += T[u]

        for v in G[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    
    return ans

print(dfs())
# %%
#ABC007 C
from collections import deque

R, C = map(int, input().split())
sy, sx = map(int, input().split())
gy, gx = map(int, input().split())
sy, sx, gy, gx = sy-1, sx-1, gy-1, gx-1
c = [[c for c in input()] for _ in range(R)]
visited = [[-1]*C for _ in range(R)]

def bfs(sy, sx, gy, gx, c, visited):
    visited[sy][sx] = 0
    Q = deque([])
    Q.append([sy, sx])
    while Q:
        y, x = Q.popleft()

        if [y, x] == [gy, gx]:
            return visited[y][x]
        
        for i, j in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            if c[y+i][x+j] == '.' and visited[y+i][x+j] == -1:
                visited[y+i][x+j] = visited[y][x]+1
                Q.append([y+i,x+j])

print(bfs(sy, sx, gy, gx, c, visited))
# %%
#ITP1_7_B
n, x = map(int, input().split())
cnt = 0

while n != 0:
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                if (i+1)+(j+1)+(k+1) == x:
                    cnt += 1
    print(cnt)
    n, x = map(int, input().split())
    cnt = 0
# %%
#ABC106 B
N = int(input())
ans = 0
for i in range(1, N+1,2):
    cnt = 0
    for j in range(1, N+1, 2):
        if i%j == 0:
            cnt += 1
    if cnt == 8:
        ans += 1

print(ans)
# %%
#ABC159 B
S = list(input())
N = len(S)
if S[:(N-1)//2] == list(reversed(S[:(N-1)//2])) and S[(N+1)//2:] == list(reversed(S[(N+1)//2:])) and S == list(reversed(S)):
  print('Yes')
else:
  print('No')
# %%
#ABC128 C
N, M = map(int, input().split())
s = [list(map(lambda x:int(x)-1,input().split()))[1:] for _ in range(M)]
p = list(map(int, input().split()))
k = [0]*M
ans = 0

for i in range(1<<N):
    for m in range(M):
        on_sum = 0
        for j in range(N):
            if (i>>j) & 1 and j in s[m]:
                on_sum += 1
        if on_sum % 2 != p[m]:
            break
    else:
        ans += 1
print(ans)
# %%
#ABC007 
from collections import deque

R, C = map(int, input().split())
sy, sx = map(int, input().split())
gy, gx = map(int, input().split())
sy, sx, gy, gx = sy-1, sx-1, gy-1, gx-1
c = [[c for c in input()] for _ in range(R)]
visited = [[-1]*C for _ in range(R)]

def bfs(sy, sx, gy, gx, c, visited):
    visited[sy][sx] = 0
    Q = deque([])
    Q.append([sy, sx])
    while Q:
        y, x = Q.popleft()

        if[y, x] == [gy, gx]:
            return visited[y][x]
        
        for i, j in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if c[y+i][x+j] == '.' and visited[y+i][x+j] == -1:
                visited[y+i][x+j] = visited[y][x]+1
                Q.append([y+i, x+j])
                
print(bfs(sy, sx, gy, gx, c, visited))
# %%
#第１０回日本情報オリンピック 予選（過去問） E
from collections import deque

H, W, N = map(int, input().split())
mymap = [[c for c in input()] for _ in range(H)]
visited = [[-1]*W for _ in range(H)]


sy, sx = map(int, input().split())
gy, gx = map(int, input().split())
sy, sx, gy, gx = sy-1, sx-1, gy-1, gx-1

# %%
#ABC006 B
#DPでの実装
def trib_dynamic(n):
    #結果を保持する辞書
    cal_result = {}

    #初期値の設定
    cal_result[0] = 0
    cal_result[1] = 0
    cal_result[2] = 0
    cal_result[3] = 1

    if n == 1 or n == 2:
        return 0
    elif n == 3:
        return 1
    
    for i in range(4, n+1):
        cal_result[i] = (cal_result[i-1] + cal_result[i-2] + cal_result[i-3])%10007
    
    return cal_result[n]

n = int(input())

print(trib_dynamic(n))
# %%
#ABC006 B
#メモ化での実装
class trib_memo:
    def __init__(self):
        self.trib_memo = {}
    
    def cal_trib(self, n):
        if n == 1 or n == 2:
            self.trib_memo[n] = 0
            return 0
        elif n == 3:
            self.trib_memo[n] = 0
            return 1
        
        if n in self.trib_memo.keys():
            return self.trib_memo[n]
        
        self.trib_memo[n] = self.cal_trib(n-1) + self.cal_trib(n-2) + self.cal_trib(n-3)
        return self.trib_memo[n]

n = int(input())
print(trib_memo(n))
# %%
#ABC227 A
N, K, A = map(int, input().split())

if N-A+1 >= K:
    print(A+K-1)
else:
    card_1 = N-A+1
    card_A = K - card_1
    if card_A%N == 0:
        print(N)
    else:
        print(card_A%N)
# %%
#ABC227 B
N = int(input())
S = list(map(int, input().split()))
my_set = set()

for a in range(1, 142):
    b = 1
    while 4*a*b + 3*(a + b) <= 1000:
        cnt = 4*a*b + 3*(a + b)
        my_set.add(cnt)
        b += 1

count = 0
for ans in S:
    if ans not in my_set:
        count += 1

print(count)
# %%
#ABC227 C
from math import floor

N = int(input())
ans = N

for i in range(2, floor(N**0.5)+1):
    ans += N//i - i + 1

for i in range(2, floor(N**1/3)+1):
    for j in range(i, floor((N/i)**0.5)+1):
        ans += floor(N/(i*j)) - j + 1

print(ans)
# %%
#ABC223 A
X = int(input())
if X != 0 and X%100 == 0:
    print('Yes')
else:
    print('No')
# %%
#ABC223 B
S = input()
S_list = []

def rolling(str , n):
    return str[n:len(str)] + str[:n]

for i in range(len(S)):
    S_list.append(rolling(S, i))

print(min(S_list))
print(max(S_list))

# %%
#ABC223 C
N = int(input())
L = []
sec = 0
for _ in range(N):
    a, b = map(int, input().split())
    L.append((a, b))
    sec += a/b

rem = sec/2
ans = 0

for a, b in L:
    if rem >= a/b:
        ans += a
        rem -= a/b
    else:
        ans += rem * b
        break

print(ans)
# %%
#ABC223 D
import queue

N, M = map(int, input().split())

G = [[] for _ in range(N+1)]
cnt = [0] * (N+1)
for _ in range(M):
    a, b = map(int, input().split())
    cnt[b] += 1
    G[a].append(b)

ans = []
pq = queue.PriorityQueue()

for i in range(1, N+1):
    if cnt[i] == 0:
        pq.push(i)

while pq:
    u = pq.pop()
    ans.append(u)
    for v in G[u]:
        cnt[v] -= 1
        if cnt[v] == 0:
            pq.push(v)

if len(ans) == N:
    print(*ans)
else:
    print(-1)
# %%
#ABC222 A
N = int(input())
ans = format(N, '04')
print(ans)
# %%
#ABC222 B
N, P = map(int, input().split())
a = list(map(int, input().split()))

ans = sum(x<P for x in a)
print(ans)
# %%
#ABC222 C
N, M = map(int, input().split())
A = [input() for i in range(2*N)]
rank = [[0,i] for i in range(2*N)]

def judge(a, b):
    if a==b: return -1
    if a=='G' and b=='P': return 1
    if a=='C' and b=='G': return 1
    if a=='P' and b=='C': return 1
    return 0

for j in range(M):
    for i in range(N):
        player1 = rank[2*i][1]
        player2 = rank[2*i+1][1]
        result = judge(A[player1][j], A[player2][j])
        if result != -1:
            rank[2*i+result][0] -= 1
    rank.sort()

for _, i in rank:print(i+1)
# %%
#ABC221 C
N_ = sorted(input(), reverse=True)
N = len(N_)
ans = 0

for i in range(1<<N):
    l = ''
    r = ''
    for j in range(N):
        if (i>>j) & 1:
            l += N_[j]
        else:
            r += N_[j]

        if l != '' and r != '' and l[0] != 0 and r[0] != 0:
            ans = max(int(l)*int(r), ans)

print(ans)
# %%
#ABC227 C
N = int(input())
ans = 0

for i in range(1, N+1):
    if i*i*i>N:
        break
    for j in range(i, N+1):
        if i*j*j>N:
            break
        ans += N//i//j-j+1

print(ans)
# %%
#ABC226 D
from math import gcd

N = int(input())
P = [list(map(int, input().split())) for _ in range(N)]
ans_list = set()

for i in range(N):
    for j in range(N):
        delta_x = P[j][0]-P[i][0]
        delta_y = P[j][1]-P[i][1]
        mygcd = gcd(delta_x, delta_y)
        if delta_x != 0 or delta_y != 0:
            ans_list.add((delta_x//mygcd, delta_y//mygcd))

print(len(ans_list))
# %%
def main():
    N, Q = map(int, input().split())
    _next = [-1] * (N + 1)  # すぐ後ろの電車の番号
    _prev = [-1] * (N + 1)  # すぐ前の電車の番号

    for _ in range(Q):
        query = list(map(int, input().split()))
        q = query[0]
        if q == 1:
            x, y = query[1:]
            _next[x] = y  # xの後ろにyがくる
            _prev[y] = x  # yの前にxがくる
        elif q == 2:
            x, y = query[1:]
            _next[x] = -1  # xが新しい最後尾
            _prev[y] = -1  # yが新しい先頭
        else:
            x = query[1]
            ans = []

            curr = x

            # まず電車を先頭までたどります
            while _prev[curr] != -1:
                curr = _prev[curr]

            # 先頭から最後尾までたどっていきます
            while curr != -1:
                ans.append(curr)
                curr = _next[curr]

            print(len(ans), *ans)  # 連結成分の大きさも出力することに注意


if __name__ == '__main__':
    main()
# %%
#ABC220 A
A, B, C = map(int, input().split())

for i in range(A, B+1):
    if i%C == 0:
        print(i)
        exit()

print(-1)
# %%
#ABC220 B
K = int(input())
A, B = map(int, input().split())

def Base_n_to_10(X,n):
    out = 0
    for i in range(1,len(str(X))+1):
        out += int(X[-i])*(n**(i-1))
    return out #int out

A_10 = Base_n_to_10(str(A), K)
B_10 = Base_n_to_10(str(B), K)

print(A_10*B_10)
# %%
#ABC220 C
N = int(input())
A = list(map(int, input().split()))
X = int(input())
sumA = sum(A)
ans = (X//sumA)*N
B = X%sumA

for i in A:
    B -= i
    ans += 1
    if B < 0:
        print(ans)
        break
# %%
#ABC219 B
S1 = input()
S2 = input()
S3 = input()
T = list(input())
mydict = {'1':S1, '2':S2, '3':S3}
ans = ''

for digit in T:
    ans += mydict[digit]

print(ans)
# %%
#ABC219 C
import copy

X = list(input())
alphabet = list('abcdefghijklmnopqrstuvwxyz')
N = int(input())
S = [input() for _ in range(N)]
S_ = ['']*N
mydict = dict(zip(X, alphabet))
mydict_ = dict(zip(alphabet, X))

for i in range(N):
    for j in list(S[i]):
        S_[i] += mydict[j]

S_.sort()
S = ['']*N

for i in range(N):
    for j in list(S_[i]):
        S[i] += mydict_[j]
    print(S[i])
# %%
#ABC218 B
P = list(map(int, input().split()))
ans = ''
alphabets = list('abcdefghijklmnopqrstuvwxyz')

for i in P:
    ans += alphabets[i-1]

print(ans)
# %%
#ABC218 D
from collections import defaultdict

N = int(input())
p=[]
p_exist = defaultdict(int)

for i in range(N):
    x,y=map(int, input().split())
    p.append([x,y])
    p_exist[(x,y)]=1

ans = 0

for p1 in range(N):
    for p2 in range(p1+1, N):
        p1_x, p1_y = p[p1]
        p2_x, p2_y = p[p2]

        if p1_x==p2_x or p1_y==p2_y:
            continue

        if p_exist[(p1_x, p2_y)] == 1 and p_exist[(p2_x, p1_y)] == 1:
            ans += 1

ans//=2
print(ans)
# %%
#ABC218 C
import numpy as np

N = int(input())
S = np.array([list(input()) for _ in range(N)])
T = np.array([list(input()) for _ in range(N)])

if np.count_nonzero(S == '#') != np.count_nonzero(S == '#'):
    print('No')


# %%
#ABC220 D
MOD = 998244353

N = int(input())
A = list(map(int, input().split()))

dp = [[0] * 10 for _ in range(N)]
dp[0][A[0]] = 1

for i in range(1, N):
    for j in range(10):
        f = (j + A[i]) % 10
        g = (j * A[i]) % 10
        dp[i][f] += dp[i - 1][j]
        dp[i][g] += dp[i - 1][j]
        dp[i][f] %= MOD
        dp[i][g] %= MOD

for i in range(10):
    print(dp[N-1][i])
# %%
#ABC217 C
N = int(input())
p = list(map(int, input().split()))
ans = [0]*N

for i in range(N):
    ans[p[i]-1] = i+1

print(*ans)
# %%
#ABC216 B
N = int(input())
full_name = [list(map(str, input().split())) for _ in range(N)]

for i in range(N):
    for j in range(i+1, N):
        if i == j:
            continue
        if full_name[i] == full_name[j]:
            print('Yes')
            exit()

print('No')
# %%
#ABC216 C
N = int(input())
ans = ''

while N != 0:
    if N%2 == 0:
        N //= 2
        ans += 'B'
    else:
        N -= 1
        ans += 'A'

ans = ''.join(list(reversed(ans)))
print(ans)
# %%
#ABC228 A
S, T, X = map(int, input().split())
if S <= T:
    if S <= X < T:
        print('Yes')
    else:
        print('No')
else:
    if 0 <= X < T or S <= X:
        print('Yes')
    else:
        print('No') 
# %%
#ABC228 B
N, X = map(int, input().split())
A = list(map(int, input().split()))
ans = 1
next = X-1
known = set()
known.add(X-1)

for i in range(N):
    next = A[next]-1
    if not next in known:
        known.add(next)
        ans += 1
    else:
        print(ans)
        break
# %%
#ABC228 C
import numpy as np

N, K = map(int, input().split())
Psum = [0]*N

for i in range(N):
    Psum[i] = sum(map(int, input().split()))

Psum_ = sorted(Psum,reverse=True)
thr = Psum_[K-1]

for i in range(N):
    if Psum[i]+300 >= thr:
        print('Yes')
    else:
        print('No')

# %%
#ABC228 D
N = 2 ** 20
Q = int(input())
A = [-1] * N
P = list(range(N))

for _ in range(Q):
    t, x = map(int, input().split())
    if t == 1:
        h = x % N
        pos = h
        visited = [pos]
        while A[pos] != -1:
            pos = P[pos]
            visited.append(pos)
        A[pos] = x
        new_P = P[(pos + 1) % N]
        for u in visited:
            P[u] = new_P
    else:
        print(A[x % N])
# %%
#ABC215 B
N = int(input())
i = 0
while 2**i <= N:
    i += 1
print(i-1)
# %%
#ABC215 C
from itertools import permutations

S, K = map(str, input().split())
K = int(K)

lst = list(set(''.join(p) for p in permutations(S)))
lst.sort()
print(lst[K-1])
# %%
#ABC138_D
import sys
input = sys.stdin.readline
from collections import deque

N, Q = map(int, input().split())
cnt = [0]*(N+1)
G = [[] for _ in range(N+1)]

for _ in range(N-1):
    a, b = map(int, input().split())
    G[a].append(b)
    G[b].append(a)

for _ in range(q):
    v, val = map(int, input().split())
    cnt[v] += val

q = deque()
q.append(1)
visited = [0]*(N+1)

while q:
    v = q.pop()
    visited[v] = 1
    for u in G[v]:
        if visited[u] == 1:
            continue
        cnt[u] += cnt[v]
        q.append(u)

print(*cnt[1:])
# %%
#第８回日本情報オリンピック 予選（過去問）
from collections import deque
import copy

m = int(input())
n = int(input())
c = [[c for c in input().split()] for _ in range(n)]
visited = [[-1]*m for _ in range(n)]
visited_ = copy.deepcopy(visited)
ans = 0

def dfs(sy, sx, c, visited_):
    global ans
    visited = copy.deepcopy(visited_)
    visited[sy][sx] = 1
    Q = deque([])
    Q.append([sy, sx])
    while Q:
        y, x = Q.popleft()
        for i, j in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if 0 <= y+i < n and 0 <= x+j < m and c[y+i][x+j] == '1' and visited[y+i][x+j] == -1:
                visited[y+i][x+j] = visited[y][x]+1
                ans = max(ans, visited[y+i][x+j])
                Q.append([y+i, x+j])
    return ans

for i in range(n):
    for j in range(m):
        if c[i][j] == '1':
            ans = max(ans,dfs(i, j, c, visited_))

print(ans)     
# %%
#ABC223 D
#優先度付きキュー heapq
import heapq

N, M = map(int, input().split())
G = [[] for _ in range(N+1)]
cnt = [0] * (N+1)

for _ in range(M):
    A, B = map(int, input().split())
    cnt[B] += 1 #縛る条件の数
    G[A].append(B)

q = []

for i in range(1, N+1):
    if cnt[i] == 0:
        q.append(i)

heapq.heapify(q)

ans=[]

while 0<len(q):
    now = heapq.heappop(q)
    ans.append(now)

    for to in G[now]:
        cnt[to] -= 1
        if cnt[to] == 0:
            heapq.heappush(q, to)

if len(ans) == N:
    print(*ans)
else:
    print(-1)
# %%
#ABC222 D
#DP 累積和使わないと間に合わない
N = int(input())
A=[0]+list(map(int, input().split()))
B=[0]+list(map(int, input().split()))
M = 3000
mod = 998244353
dp = [[0]*3001 for i in range(N+1)]

for x in range(A[1], B[1]+1):
    dp[1][x] = 1

for i in range(2, N+1):
    cum_sum = [0]*3001
    cum_sum[0] = dp[i-1][0]
    for k in range(1, 3001):
        cum_sum[k]=cum_sum[k-1]+dp[i-1][k]
        cum_sum[k]%=mod
    
    for x in range(3001):
        if A[i]<=x<=B[i]:
            dp[i][x]=cum_sum[x]

ans=sum(dp[N])
ans%=mod
print(ans)
# %%
#ABC229 A
S = [input() for _ in range(2)]
if S == ['#.', '.#'] or S == ['.#', '#.']:
    print('No')
else:
    print('Yes')
# %%
#ABC229 B
A, B = map(str, input().split())
A = A.zfill(19)
B = B.zfill(19)
flag = 0

for i in range(19):
    if int(A[i])+int(B[i]) >= 10:
        flag = 1

if flag:
    print('Hard')
else:
    print('Easy')
# %%
#ABC229 C
N, W = map(int, input().split())
AB = [tuple(map(int, input().split())) for _ in range(N)]
AB.sort(reverse=True)
ans = 0

for i in range(N):
    if W - AB[i][1] > 0:
        W -= AB[i][1]
        ans += AB[i][0]*AB[i][1]
    else:
        ans += AB[i][0]*W
        break

print(ans)
# %%
#ABC229 D
#しゃくとり法　尺取り法
from collections import deque

S = input()
K = int(input())
ans = 0

left = 0
right = 0
q = deque()

for chara in S:
    if chara == '.':
        K -= 1
    ans = max(ans, len(q))
    q.append(chara)

    while K < 0:
        flag = q.popleft()
        if flag == '.':
            K += 1

ans = max(ans, len(q))

print(ans)
# %%
#ABC221 D
from collections import Counter

N = int(input())
C = Counter()
for _ in range(N):
    a, b = map(int, input().split())
    C[a] += 1
    C[a+b] -= 1

ans = [0]*(N+1)
days = sorted(C.keys())
prev_day = 0
cnt = 0
for curr_day in days:
    ans[cnt] +=  curr_day - prev_day
    cnt += C[curr_day]
    prev_day = curr_day

print(*ans[1:])
# %%
#ABC218 C
N = int(input())

def read():
    S = set()
    for y in range(N):
        l = input()
        for x in range(N):
            if l[x] == '#':
                S.add((x, y))
    return S

S = read()
T = read()

for _ in range(4):
    mx, my = min(S)
    S = set((x-mx, y-my) for x, y in S)
    mx, my = min(T)
    T = set((x-mx, y-my) for x, y in T)

    if S==T:
        print('Yes')
        exit(0)
    
    T = set((y, -x) for x, y in T)

print('No')
# %%
#ABC217 D
#平衡二分探索木　出来るならC++で書け
#arrayで二分探索
def main():
    import sys
    readline = sys.stdin.readline
    from array import array
    import bisect

    L, Q = map(int, readline().split())
    arr = array('i', [0, L])

    for _ in range(Q):
        c, x = map(int, readline().split())
        if c == 1:
            bisect.insort_left(arr, x)
        else:
            ind = bisect.bisect_left(arr, x)
            print(arr[ind] - arr[ind - 1])


if __name__ == '__main__':
    main()
# %%
#ABC216 D
#閉路検出 DAG トポロジカルソート
from collections import deque

N, M = map(int, input().split())
G = [[] for _ in range(N+1)]
cnt = [0] * (N+1)

for _ in range(M):
    k = int(input())
    a = list(map(int, input().split()))
    for i in range(k-1):
        G[a[i]].append(a[i+1])
        cnt[a[i+1]] += 1

L = [] #トポロジカルソートの結果が入るリスト
S = deque([i for i in range(1, N+1) if cnt[i] == 0])

while S:
    u = S.popleft()
    L.append(u)
    for v in G[u]:
        cnt[v] -= 1
        if cnt[v] == 0:
            S.append(v)

if len(L) == N:
    print('Yes')
else:
    print('No')
# %%
#ABC215 D
#エラトステネスの篩
N, M = map(int, input().split())
A = list(map(int, input().split()))
maxA = max(max(A), M)

k = [True]*(maxA+1)
isprime = [True] * (maxA+1)
prime = []

for a in A:
    k[a] = False

for i in range(2, maxA+1):
    if not isprime[i]: #Falseの場合continue
        continue
    for j in range(i*2, maxA+1, i):
        isprime[j] = False
        k[i] = k[i] and k[j]
    if not k[i]:
        prime.append(i)

for p in prime:
    for j in range(p*2, M+1, p):
        k[j] = k[j] and k[p]

ans = [1]
for i in range(2,M+1):
    if k[i]:
        ans.append(i)

print(len(ans))
for i in ans:
    print(i)
# %%
#ABC213 B
N = int(input())
A = list(map(int, input().split()))
A_ = sorted(A, reverse=True)
print(A.index(A_[1])+1)
# %%
#ABC213 C
#座標圧縮
H, W, N = map(int, input().split())
R = []
C = []

for _ in range(N):
    r, c = map(int, input().split())
    R.append(r)
    C.append(c)

Rs = sorted(set(R))
Cs = sorted(set(C))

Rd = {x: i for i, x in enumerate(Rs, 1)} # Rd = {Rs[i]: i+1 for i in range(len(Rs))} と同じ
Cd = {x: i for i, x in enumerate(Cs, 1)}

for r, c in zip(R, C):
    print(Rd[r], Cd[c])
# %%
#ABC213 D
#木のオイラーツアー
import sys
sys.setrecursionlimit(300000)

N = int(input())
G = [[] for _ in range(N+1)]

for _ in range(N-1):
    A, B = map(int, input().split())
    G[A].append(B)
    G[B].append(A)

for i in range(N+1):
    G[i].sort()

ans = []

def dfs(u, p):
    ans.append(u)
    for v in G[u]:
        if v != p:
            dfs(v, u)
            ans.append(u)

dfs(1, -1)
print(*ans)
# %%
#ABC212 B
mylist = list(input())
X1, X2, X3, X4 = [int(s) for s in mylist]

if X1 == X2 and X1 == X3 and X1 == X4:
    print('Weak')
elif (X1+3)%10 == (X2+2)%10 and (X1+3)%10 == (X3+1)%10 and (X1+3)%10 == X4%10:
    print('Weak')
else:
    print('Strong')
# %%
#ABC212 C
import heapq

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
heapq.heapify(A)
heapq.heapify(B)
a = heapq.heappop(A)
b = heapq.heappop(B)
ans = abs(a-b)

while A or B:
    if a >= b and B:
        b = heapq.heappop(B)
        ans = min(ans,abs(a-b))
    elif a < b and A:
        a = heapq.heappop(A)
        ans = min(ans,abs(a-b))
    else:
        break

print(ans)