# %%
#ABC230 C
N, A, B = map(int, input().split())
P, Q, R, S = map(int, input().split())

H = Q - P + 1
W = S - R + 1

ans = [['.'] * W for _ in range(H)]

paint = set()
for i in range(0, Q-P+1):
    paint.add((i, i-A+B+P-R))
for i in range(0, Q-P+1):
    paint.add((i, -i+A+B-P-R))

for y, x in paint:
    if 0 <= y and y <= Q-P and 0 <= x and x <= S-R:
        ans[y][x] = '#'

for i in range(H):
    print(''.join(ans[i]))
# %%
#ABC230 D
#区間スケジューリング問題
from operator import itemgetter

N, D = map(int, input().split())
LR = [list(map(int, input().split())) for _ in range(N)]
LR.sort(key=itemgetter(1))

ans = 0
x = 0

for l, r in LR:
    if x < l:
        ans += 1
        x = r + D - 1

print(ans)
# %%
#ABC212 D
#累積和
import heapq

Q = int(input())
que = list()
plus = [0]*(Q+1)
ruiseki = [0]*(Q+1)

for i in range(1, Q+1):
    query=list(map(int, input().split()))

    if query[0] == 1:
        X = query[1]
        minus = ruiseki[i-1]
        heapq.heappush(que, [X-minus, i, minus])
        plus[i] = 0
        ruiseki[i] = ruiseki[i-1]+plus[i]
    
    elif query[0] == 2:
        X = query[1]
        plus[i] = X
        ruiseki[i] = ruiseki[i-1]+plus[i]

    else:
        num,indx,minus=heapq.heappop(que)
        plus[i]=0
        ruiseki[i]=ruiseki[i-1]+plus[i]
        ad=ruiseki[i]-ruiseki[indx]
        print(num+minus+ad)
# %%
#ABC211 C
#動的計画法 DP
from collections import Counter
MOD = 10**9+7

S = input()
T = '*chokudai'
dp = Counter()
dp['*'] = 1
for char in S:
    if char in T:
        char_prev = T[T.index(char) - 1]
        dp[char] += dp[char_prev]
        dp[char] %= MOD
print(dp['i'])
# %%
#ABC211 D
#幅優先探索 BFS
from collections import deque

INF = 1  << 60
MOD = 10 ** 9 + 7

N, M = map(int, input().split())
G = [[] for _ in range(N+1)]
for _ in range(M):
    A, B = map(int, input().split())
    G[A].append(B)
    G[B].append(A)

visited = [False]*(N+1)
visited[1] = True

dist = [0]*(N+1)

count = [0]*(N+1)
count[1] = 1

que = deque()
que.append(1)

while que:
    now = que.popleft()
    for to in G[now]:
        if visited[to] == False:
            count[to] = count[now]
            count[to] %= MOD
            dist[to] = dist[now] + 1
            visited[to] = True
            que.append(to)
        else:
            if dist[to] == dist[now]+1:
                count[to] += count[now]
                count[to] %= MOD

if visited[N] == False:
    print(0)
else:
    print(count[N])
# %%
#ABC228 D
#UnionFind 経路圧縮 
Q = int(input())
N = 1048576
A = [-1] * N
P = list(range(N))

for _ in range(Q):
    t, x = map(int, input().split())

    if t == 1:
        pos = x % N
        visited = [pos]
        while A[pos] != -1:
            pos = P[pos]
            visited.append(pos)
        A[pos] = x
        new_p = P[(pos + 1) % N]
        for u in visited:
            P[u] = new_p
    
    else:
        print(A[x % N])
# %%
#ABC218 E
N, M = map(int, input().split())
G = [[] for _ in range(N+1)]

for _ in range(M):
    A, B, C = map(int, input().split())
    G[A].append(B)
    G[B].append(A)
# %%
#ARC131 A
A = input()
B = int(input()) * 5
B = str(B).zfill(9)
print(A+B)
# %%
#ARC131 B
import random

H, W = map(int, input().split())
color = set(['.', '1', '2', '3', '4', '5'])
c = []

c.append(['.']*(W+2))
for _ in range(H):
    line = list(input())
    c.append(['.']+line+['.'])
c.append(['.']*(W+2))

for h in range(1, H+1):
    for w in range(1, W+1):
        if c[h][w] == '.':
            painted = set('.')
            for y, x in (1, 0), (-1, 0), (0, 1), (0, -1):
                painted.add(c[h+y][w+x])
            use = list(color - painted)
            c[h][w] = random.choice(use)

for i in range(1, H+1):
    print(''.join(c[i][1:W+1]))
# %%
#ARC131 D
N, M, D = map(int, input().split())
center, *r = list(map(int, input().split()))
s = list(map(int, input().split()))

if N%2 == 0:
    N = N//2
    P = tuple(range(D, (N+1)*D, D))
    score = [0] * (N)
    cnt = 0
    for j in range(N):
        for i in range(cnt, M):
            if P[j] < r[i]:
                score[j] = s[i-1]
            else:
                cnt += 1
    print(2 * sum(score) + s[0] - s[-1])

else:
    N = (N-1)//2
    P = tuple(range(D, (N+1)*D, D))
    score = [0] * (N)
    cnt = 0
    for j in range(N):
        for i in range(cnt, M):
            if P[j] < r[i]:
                score[j] = s[i-1]
            else:
                cnt += 1
    print(2 * sum(score) + s[0])


# %%
#全国統一プログラミング王決定戦本戦 A
#累積和
N = int(input())
A = list(map(int, input().split()))

ruiseki = [0] * (N+1)
ruiseki[0] = 0
for i in range(1, N+1):
    ruiseki[i] = A[i-1] + ruiseki[i-1]

for i in range(1, N+1):
    ans = 0
    for j in range(i, N+1):
        ans = max(ans, ruiseki[j] - ruiseki[j-i])
    print(ans)
# %%
#第９回日本情報オリンピック 本選（過去問） A
#累積和
n, m = map(int, input().split())
ruiseki = [0]

for _ in range(n-1):
    dist = int(input())
    ruiseki.append(int(dist + ruiseki[-1]))

ans = 0
curr = 0
for i in range(m):
    move = int(input())
    prev = curr
    curr += move
    ans += abs(ruiseki[curr] - ruiseki[prev])
    ans %= 10**5

print(ans)
# %%
#第１０回日本情報オリンピック 本選（過去問） A

# %%
#ABC106 D
#二次元累積和
def main():
    import sys
    input = sys.stdin.readline
    inf = float("inf")
 
    N,M,Q = map(int, input().split())
 
    list_t = [ [0] * N for _ in range(N) ]
    for _ in range(M):
        L,R = map(int, input().split())
        list_t[L-1][R-1] += 1
 
    list_s = [ [0] * (N+1) for _ in range(N+1)]
    for i in range(N):
        for j in range(N):
            list_s[i+1][j+1] = list_s[i+1][j] + list_s[i][j+1] - list_s[i][j] + list_t[i][j]
 
    for _ in range(Q):
        p,q = map(int, input().split())
        ans = list_s[q][q] - list_s[q][p-1] - list_s[p-1][q] + list_s[p-1][p-1]
        print(ans)
 
if __name__ == "__main__":
    main()
# %%
#ABC014 C
#いもす法
n = int(input())
s = [0] * 1000002

for _ in range(n):
    a, b = map(int, input().split())
    s[a] += 1
    s[b+1] -= 1

for i in range(1, 1000001):
    s[i] += s[i-1]

print(max(s))
# %%
#JOI 2008/2009 本選 問題2
#二分探索
D = int(input())
N = int(input())
M = int(input())
 
S = [0] + [int(input()) for _ in range(N-1)] + [D]
S.sort()
 
import bisect
ans = 0
for _ in range(M):
    k = int(input())
    idx = bisect.bisect_right(S, k)
    ans += min(k - S[idx-1], S[idx] - k)
print(ans)
# %%
#ABC210 C
#しゃくとり法
from collections import Counter

N, K = map(int, input().split())
c = tuple(map(int, input().split()))
counter = Counter(c[:K])
ans = len(counter)

for i in range(K, N):
    left = c[i-K]
    right = c[i]
    counter[left] -= 1
    counter[right] += 1
    if counter[left] == 0:
        del counter[left]
    ans = max(ans, len(counter))

print(ans)
# %%
#ABC209 C
N = int(input())
C = list(map(int, input().split()))
C.sort()
ans=1
for i in range(N):
    ans = ans * max(0, C[i] - i) % 1000000007
print(ans)
# %%
#ABC209 D
from collections import deque

N, Q = map(int, input().split())
G = [[] for _ in range(N+1)]
visited = [False]*(N+1)

for _ in range(N-1):
    a, b = map(int, input().split())
    G[a].append(b)
    G[b].append(a)

que = deque()
color = [-1] * (N+1)
color[1] = 0
visited[1] = True

que.append(1)

while que:
    now = que.popleft()
    now_color = color[now]

    for to in G[now]:
        if visited[to] == False:
            visited[to] = True
            if now_color == 0:
                color[to] = 1
            if now_color == 1:
                color[to] = 0
            que.append(to)

for _ in range(Q):
    c, d = map(int, input().split())
    if color[c] == color[d]:
        print('Town')
    else:
        print('Road')
# %%
#ABC208 C
#座標圧縮
N, K = map(int, input().split())
a = list(map(int, input().split()))

a_sort = sorted(a)

num_order = dict()
for i in range(N):
    num_order[a_sort[i]] = i+1

syou = K//N
amari = K%N

for num in a:
    if num_order[num] <= amari:
        print(syou+1)
    else:
        print(syou)
# %%
#ABC208 D
#ワーシャルフロイド法
N, M = map(int, input().split())
inf = 10 ** 10
time = [[inf] * (N+1) for _ in range(N+1)]

for i in range(1, N+1):
    time[i][i] = 0

for _ in range(M):
    A, B, C = map(int, input().split())
    time[A][B] = C

ans = 0

for k in range(1, N+1):
    new_time = [[0]*(N+1) for _ in range(N+1)]

    for start in range(1, N+1):
        for goal in range(1, N+1):
            new_time[start][goal] = min(time[start][goal], time[start][k] + time[k][goal])

            if new_time[start][goal] != inf:
                ans += new_time[start][goal]
    
    time = new_time

print(ans)
# %%
#ABC231 D
#UnionFind
from typing import List
class UnionFind:
    """0-indexed"""

    def __init__(self, n):
        self.n = n
        self.parent = [-1] * n
        self.__group_count = n

    def unite(self, x, y) -> bool:
        """xとyをマージ"""
        x = self.root(x)
        y = self.root(y)

        if x == y:
            return False

        self.__group_count -= 1

        if self.parent[x] > self.parent[y]:
            x, y = y, x

        self.parent[x] += self.parent[y]
        self.parent[y] = x

        return True

    def is_same(self, x, y) -> bool:
        """xとyが同じ連結成分か判定"""
        return self.root(x) == self.root(y)

    def root(self, x) -> int:
        """xの根を取得"""
        if self.parent[x] < 0:
            return x
        else:
            self.parent[x] = self.root(self.parent[x])
            return self.parent[x]

    def size(self, x) -> int:
        """xが属する連結成分のサイズを取得"""
        return -self.parent[self.root(x)]

    def all_sizes(self) -> List[int]:
        """全連結成分のサイズのリストを取得 O(N)"""
        sizes = []
        for i in range(self.n):
            size = self.parent[i]
            if size < 0:
                sizes.append(-size)
        return sizes

    def groups(self) -> List[List[int]]:
        """全連結成分の内容のリストを取得 O(N・α(N))"""
        groups = dict()
        for i in range(self.n):
            p = self.root(i)
            if not groups.get(p):
                groups[p] = []
            groups[p].append(i)
        return list(groups.values())

    @property
    def group_count(self) -> int:
        """連結成分の数を取得 O(1)"""
        return self.__group_count

def judge():
    N, M = map(int, input().split())
    uf = UnionFind(N)
    C = [0] * N  # 人が条件に出てくる回数をカウント
    for _ in range(M):
        a, b = map(int, input().split())
        a, b = a - 1, b - 1
        if uf.is_same(a, b):  # 閉路（ループ）がないか判定
            return False
        uf.unite(a, b)
        C[a] += 1
        C[b] += 1

    for i in range(N):
        if C[i] >= 3:  # 同時に3人以上と隣り合うことはできない
            return False
    return True


print("Yes" if judge() else "No")
# %%
#ABC207 C
N = int(input())
l = [0] * N
r = [0] * N

for i in range(N):
    t, l[i], r[i] = map(int, input().split())
    if t == 2:
        r[i] -= 0.5
    elif t == 3:
        l[i] += 0.5
    elif t == 4:
        l[i] += 0.5
        r[i] -= 0.5

ans = 0
for i in range(N):
    for j in range(i+1, N):
        ans += (max(l[i],l[j]) <= min(r[i],r[j]))

print(ans)
# %%
#ABC207 C
from collections import Counter

N = int(input())
A = Counter(list(map(int, input().split())))

ans = N*(N-1)//2

for i in A:
    n = A[i]
    ans -= n*(n-1)//2

print(ans)
# %%
#ABC207 D
#UnionFind
class UnionFind:
    def __init__(self, n):
        self.n=n
        self.parent_size=[-1]*n
    def merge(self, a, b):
        x, y=self.leader(a), self.leader(b)
        if x == y: return 
        if abs(self.parent_size[x])<abs(self.parent_size[y]): x, y=y, x
        self.parent_size[x] += self.parent_size[y]
        self.parent_size[y]=x
        return 
    def same(self, a, b):
        return self.leader(a) == self.leader(b)
    def leader(self, a):
        if self.parent_size[a]<0: return a
        self.parent_size[a]=self.leader(self.parent_size[a])
        return self.parent_size[a]
    def size(self, a):
        return abs(self.parent_size[self.leader(a)])
    def groups(self):
        result=[[] for _ in range(self.n)]
        for i in range(self.n):
            result[self.leader(i)].append(i)
        return [r for r in result if r!=[]]

# 入力の受け取り
N=int(input())
A=list(map(int, input().split()))

# UnionFind　要素数10^6で初期化
Uni=UnionFind(10**6)

# 答えを格納する変数
ans=0

# i=0~N//2まで
for i in range(N//2):
    # 左側
    A_left=A[i]
    # 右側
    A_right=A[N-i-1]
    # 左側と右側が違うグループなら
    if Uni.same(A_left,A_right)==False:
        # 答えに+1
        ans+=1
        # 同じグループへ
        Uni.merge(A_left,A_right)

# 答えを出力
print(ans)
# %%
S = input()
T = input()
alphabets = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz'

K = alphabets.find(T[0]) - alphabets.find(S[0])

flag = 1

for i in range (1, len(S)):
    T_idx = alphabets.find(T[i])
    if S[i] != alphabets[T_idx - K]:
        flag = 0

if flag:
    print('Yes')
else:
    print('No')
# %%
from itertools import permutations

N, M = map(int, input().split())
G1 = [[] for _ in range(N+1)]
G2 = [[] for _ in range(N+1)]

for _ in range(M):
    A, B = map(int, input().split())
    G1[A].append(B)
    G1[B].append(A)

for _ in range(M):
    C, D = map(int, input().split())
    G2[C].append(D)
    G2[D].append(C)
# %%
from itertools import permutations

N, M = map(int, input().split())
G1 = [[] for _ in range(N)]
flag = 0

if M == 0:
    print('Yes')
    exit()

for _ in range(M):
    A, B = map(int, input().split())
    A -= 1
    B -= 1
    G1[A].append(B)
    G1[B].append(A)

seq = tuple(i for i in range(N))
s = tuple(permutations(seq))

CD = [map(int, input().split()) for _ in range(M)]
C, D = [list(i) for i in zip(*CD)]

for i in range(len(s)):
    mydict = dict(zip(seq, s[i]))
    G2 = [[] for _ in range(N)]

    for j in range(M):
        c = C[j] - 1
        d = D[j] - 1
        G2[mydict[c]].append(mydict[d])
        G2[mydict[d]].append(mydict[c])

    if G1 == G2:
        flag = 1
 
if flag:
    print('Yes')
else:
    print('No')
# %%
H, W = map(int, input().split())

place = [list(input()) for _ in range(H)]
move = [[1,0],[0,1]]
 
def dfs(x,y,now_depth):
    global depth
 
    place[y][x] = 0 # 氷を割る
 
    if now_depth > depth:
        depth += 1 # 深さの最大値を超えた場合更新する
 
    for x_move,y_move in move:
        nx = x+x_move
        ny = y+y_move
 
        if 0 <= nx < W and 0 <= ny < H and place[ny][nx] == '.':
            dfs(nx,ny,now_depth+1)
    
    place[y][x] = 1 # 元に戻す(地点を探索する前のリストの状態にする)
        
# 全てのマス目に対して可能性を試す
depth = 0
dfs(0, 0, 1)
 
print(depth)
# %%
#ABC233 D
from collections import Counter

N, K = map(int, input().split())
A = list(map(int, input().split()))

S = []
S.append(0)
cnt = Counter()
ans = 0

for i in range(N):
    S.append(A[i] + S[-1])

for x in S:
    y = x - K
    ans += cnt[y]
    cnt[x] += 1

print(ans)
# %%
#ABC232 C
from itertools import permutations

N, M = map(int, input().split())
G1 = [[False] * N for _ in range(N)]
G2 = [[False] * N for _ in range(N)]

for _ in range(M):
    A, B = map(int, input().split())
    A -= 1
    B -= 1
    G1[A][B] = G1[B][A] = True

for _ in range(M):
    C, D = map(int, input().split())
    C -= 1
    D -= 1
    G2[C][D] = G2[D][C] = True

ans = False
for p in permutations(range(N)):
    ok = True
    for i in range(N):
        for j in range(N):
            if G1[i][j] != G2[p[i]][p[j]]:
                ok = False
    if ok:
        ans = True

print("Yes" if ans else "No")
# %%
#GRL_1_C
#ワーシャルフロイド法
V, E = map(int, input().split())
G = [[float('inf')] * V for _ in range(V)]

for i in range(V):
    G[i][i] = 0

for _ in range(E):
    s, t, d = map(int, input().split())
    G[s][t] = d

for k in range(V):
    for i in range(V):
        for j in range(V):
            G[i][j] = min(G[i][j], G[i][k]+G[k][j])

flag = 0

for i in range(V):
    if G[i][i] < 0:
        flag = 1

if flag:
    print('NEGATIVE CYCLE')
else:
    for d in G:
        print(' '.join(map(str, d)).replace('inf', 'INF'))
# %%
#ABC012 D
#ワーシャルフロイド法
INF = 10**10
N, M = map(int, input().split())

dp = [[INF] * N for _ in range(N)]
for i in range (N):
    dp[i][i] = 0

for _ in range(M):
    a, b, t = map(int, input().split())
    a, b = a-1, b-1
    dp[a][b] = dp[b][a] = t

for k in range(N):
    for i in range(N):
        for j in range(N):
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])

print(min([max(d) for d in dp]))
# %%
#ABC079 D
#ワーシャルフロイド法
H, W = map(int, input().split())
dp = [[] for _ in range(10)]

for i in range(10):
    dp[i] = list(map(int, input().split()))

for k in range(10):
    for i in range(10):
        for j in range(10):
            dp[i][j] = min(dp[i][j], dp[i][k]+dp[k][j])

ans = 0

for i in range(H):
     wall = list(map(int, input().split()))
     for j in range(W):
         if wall[j] >= 0:
             ans += dp[wall[j]][1]

print(ans)
# %%
#GRL_1_A
#ダイクストラ法
from heapq import heappush, heappop

def dijkstra(s, n):
    dist = [float('inf')] * n
    hq = [(0, s)]
    dist[s] = 0
    visited = [False] * n
    while hq:
        v = heappop(hq)[1]
        visited[v] = True
        for to, cost in adj[v]:
            if visited[to] == False and dist[v] + cost < dist[to]:
                dist[to] = dist[v] + cost
                heappush(hq, (dist[to], to))
    
    return dist

v, e, r = map(int, input().split())

adj = [[] for _ in range(v)]
for i in range(e):
    s, t, d = map(int, input().split())
    adj[s].append((t, d))

d = dijkstra(r, v)

for i in d:
    print('INF' if i == float('inf') else i)
# %%
#第７回日本情報オリンピック 予選（過去問） F
#ダイクストラ法
from heapq import heappush, heappop

n, k = map(int, input().split())

def dijkstra(s, n): #(start, nodes)
    dist = [float('inf')] * n
    hq = [(0, s)] #(distance, node)
    dist[s] = 0
    visited = [False] * n
    
    while hq:
        v = heappop(hq)[1]
        visited[v] = True
        for to, cost in adj[v]:
            if visited[to] == False and dist[v] + cost < dist[to]:
                dist[to] = dist[v] + cost
                heappush(hq, (dist[to], to))
    return dist

adj = [[] for _ in range(n+1)]

for i in range(k):
    INPUT = list(map(int, input().split()))

    if INPUT[0] == 0:
        d = dijkstra(INPUT[1], n+1)[INPUT[2]]
        print(-1 if d == float('inf') else d)
    
    else:
        adj[INPUT[1]].append((INPUT[2], INPUT[3]))
        adj[INPUT[2]].append((INPUT[1], INPUT[3]))
# %%
#第１３回日本情報オリンピック 予選（過去問） E
#BFS, ダイクストラ法
#メモリ効率悪い
from heapq import heappop, heappush
from collections import deque

N, K = map(int, input().split())
G = [[] for _ in range(N+1)]

cost_dist = [list(map(int, input().split())) for _ in range(N)]

for _ in range(K):
    A, B = map(int, input().split())
    G[A].append(B)
    G[B].append(A)

def bfs(s, N, G): #(start, nodes, map)
    visited = [False] * (N+1)
    dist = [-1] * (N+1)
    
    Q = deque()
    Q.append(s)
    visited[s] = True
    dist[s] = 0

    while Q:
        now = Q.popleft()        
        for to in G[now]:
            if visited[to] == False:
                visited[to] = True
                dist[to] = dist[now] + 1
                Q.append(to)
    
    return dist

adj = [[] for _ in range(N+1)]

for i in range(1, N+1):
    dist = bfs(i, N, G)
    for j in range(1, N+1):
        if dist[j] >= 1 and dist[j] <= cost_dist[i-1][1]:
            adj[i].append((j, cost_dist[i-1][0]))

def dijkstra(s, n): #(start, nodes)
    dist = [float('inf')] * n
    hq = [(0, s)] #(distance, node)
    dist[s] = 0
    visited = [False] * n
    
    while hq:
        v = heappop(hq)[1]
        visited[v] = True
        for to, cost in adj[v]:
            if visited[to] == False and dist[v] + cost < dist[to]:
                dist[to] = dist[v] + cost
                heappush(hq, (dist[to], to))
    return dist

d = dijkstra(1, N+1)
print(d[N])
# %%
#ABC234 D
from collections import deque
from heapq import heapify, heappop, heappush

N, K = map(int, input().split())
Q = deque(list(map(int, input().split())))

hq = []

for _ in range(K):
    heappush(hq, Q.popleft())

print(hq[0])

for i in range(K, N):
    a = Q.popleft()
    heappush(hq, a)
    heappop(hq)
    print(hq[0])
# %%
#第１０回日本情報オリンピック 予選（過去問） E
#BFS, 幅優先探索
from collections import deque

H, W, N = map(int, input().split())
c = [list(input()) for _ in range(H)]

def bfs(sh, sw, gh, gw, c, H, W):
    dist = [[-1] * W for _ in range(H)]
    dist[sh][sw] = 0
    Q = deque([])
    Q.append([sh, sw])
    while Q:
        h, w = Q.popleft()

        for dh, dw in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if not (0 <= h+dh < H and 0 <= w+dw < W) or c[h+dh][w+dw] == 'X' or dist[h+dh][w+dw] != -1:
                continue

            dist[h+dh][w+dw] = dist[h][w] + 1
            Q.append([h+dh, w+dw])
    
    return dist[gh][gw]

cheese = [[] for _ in range(N+1)]

for h in range(H):
    for w in range(W):
        if c[h][w] == 'S':
            cheese[0] = [h, w]
        
        elif c[h][w].isdecimal():
            cheese[int(c[h][w])] = [h, w]

ans = 0

for i in range(N):
    sh, sw = cheese[i]
    gh, gw = cheese[i+1]
    ans += bfs(sh, sw, gh, gw, c, H, W)

print(ans)