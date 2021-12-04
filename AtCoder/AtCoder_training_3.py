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
N, D = map(int, input().split())
cnt = 0

wall = [list(map(int, input().split())) for l in range(N)]
wall.sort(reverse=True, key = lambda x:x[1])

while wall:
    cnt += 1
    destroyed = wall[-1][1]+D-1
    idx = wall.find