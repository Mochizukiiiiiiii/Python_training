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
# %%
N = int(input())
S = list(map(int, input().split()))
T = int(input())
S_t = []

for i in range(N):
    S_t.append(S[i]//T)

print(len(set(S_t)))
# %%
N = int(input())


# %%
import numpy as np
N = int(input())

ans = np.zeros((N, N), dtype = object)
ans[:, :] = '.'

if N == 2:
    ans[0, 0] = '#'

elif N%2 == 0:
    ans[0:2, 0] = '#'
    for i in range(1, N//2):
        ans[2*i+1, :2*i+1] = '#'
        ans[:2*i+2, 2*i] = '#'
    ans[0,:] = '.'

else:
    for i in range((N+1)//2):
        ans[2*i, :2*i] = '#'
        ans[:2*i+1, 2*i-1] = '#'
    ans[0,:] = '.'

ans = ans.tolist()

for i in range(N):
    print(''.join(ans[i]))
# %%
N = int(input())
gacha = list(map(int, input().split())) #ガチャ
coin = list(map(int, input().split())) #コイン
count = 0 #正ならコインを持っている、負ならガチャをスルーしている
current_place = 0
ans = 0

for i in range(2*N):
    if count == 0: #コイン無し、ガチャスルー無し
        if gacha[0] < coin[0]:
            ans += abs(gacha[0] - current_place)
            current_place = gacha[0]
            count -= 1
            del gacha[0]
        else:
            ans += abs(coin[0] - current_place)
            current_place = coin[0]
            count += 1
            del coin[0]

    elif count > 0: #コイン有り
        if len(coin) != 0: #コインを拾いきっていない
            if gacha[0] < coin[0]:
                ans += abs(gacha[0] - current_place)
                current_place = gacha[0]
                count -= 1
                del gacha[0]
            else:
                ans += abs(coin[0] - current_place)
                current_place = coin[0]
                count += 1
                del coin[0]
        else:
            ans += abs(gacha[0] - current_place)
            current_place = gacha[0]
            count -= 1

    else: #コイン無し、ガチャスルー有り
        ans += abs(coin[0] - current_place)
        current_place = coin[0]
        count += 1
        del coin[0]

print(ans)
# %%
#ABC125 B
N = int(input())
V = list(map(int, input().split()))
C = list(map(int, input().split()))
P = []

for i in range(N):
    P.append(max(0, (V[i]-C[i])))

print(sum(P))
# %%
#ABC124 B
N = int(input())
H = list(map(int, input().split()))
max = 0
ans = 0

for i in range(N):
    if H[i] >= max:
        ans += 1
        max = H[i]

print(ans)
# %%
#ABC123 B
s = [int(input()) for i in range(5)]
S = []
for i in s:
    if i % 10 == 0:
        S.append(0)
    else:
        S.append(10 - (i % 10))
print(sum(s) + sum(S) - max(S))
# %%
#ABC119 B
N = int(input())
ans = 0

for i in range(N):
    a_, b=input().split()
    a = float(a_)

    if b == 'JPY':
        ans += a
    else:
        ans += a*380000

print(ans)
# %%
#ABC118 B
N, M = map(int, input().split())
conditions = [list(map(lambda x:int(x)-1, input().split()))[1:] for i in range(N)]
ans = 0

for i in range(M):
    count = 0
    for j in range(N):
        if i in conditions[j]:
            count += 1
    if count == N:
        ans += 1

print(ans)
# %%
#ABC117 B
N = int(input())
L = list(map(int, input().split()))

L_max = max(L)
L_sum = sum(L)

if L_max < (L_sum - L_max):
    print('Yes')
else:
    print('No')
# %%
#ABC116 B
s = int(input())
check = set()
count = 0

while not s in check:
    check.add(s)
    count += 1
    if s%2 == 0:
        s = s//2
    else:
        s = s*3+1

print(count+1)
# %%
#ABC115 B
N = int(input())
p = [int(input()) for i in range(N)]
p.sort()

print(sum(p) - p[N-1]//2)
# %%
#ABC022 A
N, S, T = map(int, input().split())
W = int(input())
A = [int(input()) for i in range(N-1)]
ans = 0
if S <= W <= T:
    ans += 1

for i in range(N-1):
    W += A[i]
    if S <= W <= T:
        ans += 1

print(ans)
# %%
#ABC079 C
ABCD = input()
A = int(ABCD[0])
B = int(ABCD[1])
C = int(ABCD[2])
D = int(ABCD[3])
list = [B, C, D]
op = ['+', '+', '+']

for i in range(1<<3):
    SUM = [0, 0, 0]
    DIFF = [0, 0, 0]
    for j in range(3):
        if ((i >> j) & 1):
            SUM[j] = list[j]
        else:
            DIFF[j] = list[j]
    ans = A + sum(SUM) - sum(DIFF)
    if ans == 7:
        ans_SUM = SUM

for i in range(3):
    if ans_SUM[i] == 0:
        op[i] = '-'

print(str(A)+op[0]+str(B)+op[1]+str(C)+op[2]+str(D)+'=7')
# %%
#square869120Contest #6 B
N = int(input())
AB = [list(map(int, input().split())) for AB in range(N)]

ans = 10**15
for i in range(N):
    for j in range(N):
        s, t = AB[i][0], AB[j][1]
        tmp = 0

        for a, b in AB:
            tmp += abs(a - s) + abs(b - a) + abs(t - b)

        ans = min(ans, tmp)

print(ans)
# %%
#第７回日本情報オリンピック 予選（過去問） D
import copy

m = int(input())
mxy = [list(map(int, input().split())) for mxy in range(m)]
n = int(input())
nxy = [list(map(int, input().split())) for nxy in range(n)]

mxy0 = mxy[0]
for i in range(n):
    count = 0
    mxy_ = copy.deepcopy(mxy)
    movex = nxy[i][0] - mxy0[0]
    movey = nxy[i][1] - mxy0[1]
    for j in range(m):
        mxy_[j][0] += movex
        mxy_[j][1] += movey
        if mxy_[j] in nxy:
            count += 1
    if count == m:
        ansx, ansy = movex, movey

print(str(ansx)+' '+str(ansy))
# %%
#ABC002 D
import itertools

N, M = map(int, input().split())
relations = [[0] * (N+1) for i in range(N+1)]
for i in range(M):
	x, y = map(int, input().split())
	relations[x][y] = 1
	relations[y][x] = 1

result = 0
for i in range(1<<N):
    group = []
    for j in range(N):
        if (i>>j) & 1:
            group.append(j + 1)
    flag = 1
    for k in itertools.combinations(group, 2):
        if relations[k[0]][k[1]] == 0:
            flag = 0
            break
    if flag == 1:
        result = max(result, len(group))

print(result)
# %%
#ABC104 A
R = int(input())
if R < 1200:
    print('ABC')
elif R < 2800:
    print('ARC')
else:
    print('AGC')
# %%
#ABC104 B
import re

S = input()
moji_cnt = len(S)
flag = 1

if S[0] != 'A':
    flag = 0

if S[2 : moji_cnt-1].count('C') != 1:
    flag = 0

if len(re.findall(r"[A-Z]", S)) != 2:
    flag = 0

if flag:
    print('AC')
else:
    print('WA')
# %%
#ABC104 C
D, G = map(int, input().split())
pc = [map(int, input().split()) for _ in range(D)]
p, c = [list(i) for i in zip(*pc)]
ans = float('inf')

for i in range(1<<D):
    total = 0
    num = 0
    for j in range(D):
        if (i >> j) & 1:
            total += 100 * (j + 1) * p[j] + c[j]
            num += p[j]
    if total >= G:
        ans = min(ans, num)
    else:
        for k in reversed(range(D)):
            if (i >> k) & 1:
                continue
            target = G - total
            if target <= p[k] * (100 * (k+1)):
                num += -(-target // (100* (k+1)))
                ans = min(ans, num)
                break
            else:
                total += 100 * (k+1) * p[k]
                num += p[k]

print(ans)
# %%
#ABC156 B
N, K = map(int, input().split())
count = 1
while K**count - 1 < N:
  count += 1

print(count)
# %%
#ABC155 B
N = int(input())
A = list(map(int, input().split()))
flag = 1

for i in range(N):
    if A[i]%2 == 0:
        if A[i]%3 != 0:
            if A[i]%5 != 0:
                flag = 0

if flag:
    print('APPROVED')
else:
    print('DENIED')
# %%
#ABC154 B
S = len(input())
X = 'x'*S
print(X)
# %%
#ABC153 B
H, N = map(int, input().split())
A = list(map(int, input().split()))

if H <= sum(A):
    print('Yes')
else:
    print('No')
# %%
#幅優先探索

#クラスを宣言
class Node:
    #コンストラクタを宣言
    def __init__(self, index):
        #メソッドを定義
        self.index = index #Nodeの番号を定義
        self.nears = [] #隣接Nodeのリストを定義
        self.sign = False #探索済みかどうか定義

    def __repr__(self):
        return f'Node index:{self.index} Node nears:{self.nears} Node sign:{self.sign}'

# %%
#ABC168 D
from collections import deque
#幅優先探索

#クラスを宣言
class Node:
    #コンストラクタを宣言
    def __init__(self, index):
        #メソッドを定義
        self.index = index #Nodeの番号を定義
        self.nears = [] #隣接Nodeのリストを定義
        self.sign = -1 #探索済みかどうか定義

    def __repr__(self):
        return f'Node index:{self.index} Node nears:{self.nears} Node sign:{self.sign}'

#入力読み込み
n, m = map(int, input().split())
links = [list(map(int, input().split())) for _ in range(m)]

#インスタンス(Node)を生成し、nodesに格納する。
# ノード 0 も生成されるが使用しない。
nodes = []
for i in range(n + 1):
    nodes.append(Node(i))

#この時点で探索済みのnodeは存在しないためsignメソッドで-1が返される。
#print([node.sign for node in nodes])

#隣接nodeをnearsメソッドに格納する。
for j in range(m):
    edge_start, edge_end = links[j]
    nodes[edge_start].nears.append(edge_end)
    nodes[edge_end].nears.append(edge_start) # 有向グラフの場合は消す


#BFS
#探索対象nodeをqueueに入れる。
queue = deque()
# 本問では node 1 から探索を開始するため queue に node 1 を最初に入れる。
queue.append(nodes[1])
#queueがなくなるまで探索を続ける。
while queue:
    #queue から node を 1 つ取り出す。取り出したノードについて調べる。
    #取り出された node は queue から消える。
    node = queue.popleft() # .pop() にすると DFS になる
    # print(node) # コメントアウトを外すと現在地がわかる。 DFS と BFS で比べてみるとよい
    # 取り出された node の隣接 node 達を nears に入れる。
    nears = node.nears
    # 隣接 node 達が探索済みか 1 つずつ調べる。
    for near in nears:
        # 未探索の隣接 node は queue に追加する
        # 取り出してきた親 node は道しるべとなるため、子 node の sign メソッドに追加する。
        if nodes[near].sign == -1:
            queue.append(nodes[near])
            nodes[near].sign = node.index

#YesまたはNoを表示
if -1 in [node.sign for node in nodes][2:]:
    print('No')
    exit(0)
else:
    print('Yes')

#道しるべを表示
for k in range(2, n+1):
    print(nodes[k].sign)
# %%
#ABC069 C
N = int(input())
a = list(map(int, input().split()))
a_4 = [i for i in a if i%4 == 0]
a_2 = [i for i in a if i%4 != 0 and i%2 == 0]

if N%2 == 0:
    if len(a_4)*2 + len(a_2) >= N:
        print('Yes')
    else:
        print('No')

else:
    if len(a_4)*2 + 1 >= N:
        print('Yes')
    elif len(a_4)*2 + len(a_2) >= N:
        print('Yes')
    else:
        print('No')
# %%
#ABC069 D
H, W = map(int, input().split())
N = int(input())
a = list(map(int, input().split()))
paint_list = []
for i in range(N):
    paint_list += ([i+1]*a[i])

def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]

painted = convert_1d_to_2d(paint_list, W)

for i in range(H):
    if i%2 != 0:
        painted[i].reverse()
        print(*painted[i])
    if i%2 == 0:
        print(*painted[i])
# %%
#ABC072 C
from collections import Counter

N = int(input())
a = list(map(int, input().split()))
l = []

for i in range(N):
    l.append(a[i]-1)
    l.append(a[i])
    l.append(a[i]+1)

ll = Counter(l)
print(ll.most_common()[0][1])
# %%
#ABC072 D
N = int(input())
p = list(map(int, input().split()))
flag = []
count = 0
seq_count = 0

for i in range(N):
    if i == p[i]-1:
        count += 1
        flag.append(str(1))
    else:
        flag.append(str(0))

flag_line = ''.join(flag)
seq_count = flag_line.count('11')

print(count-seq_count)
# %%
#ALDS_5_A
n = int(input())
A = list(map(int, input().split()))
q = int(input())
m = list(map(int, input().split()))
sum_list = []

for i in range(1<<n):
    sum = 0
    for j in range(n):
        if (i>>j) & 1:
           sum += A[j]
    sum_list.append(sum)
for i in range(q):
    if m[i] in sum_list:
        print('yes')
    else:
        print('no')
# %%
#第７回日本情報オリンピック 予選（過去問） E
R, C = map(int, input().split())
senbei = [list(map(int, input().split())) for i in range(R)]

for i in range(1<<R):
    for j in range(R):
        if (i>>j) & 1:
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
import sys
n = int(input())
tri = [0, 0, 1]

if n == 1 or n == 2:
    print(0)
    sys.exit(0)
elif n == 3:
    print(1)
    sys.exit(0)
else:
    for i in range(3, n):
        tri.append(tri[-3]+tri[-2]+tri[-1])
        del tri[0]

print(tri[-1]%10007)
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
