# %%
#ABC223 D
#優先度付きキュー　heapq
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