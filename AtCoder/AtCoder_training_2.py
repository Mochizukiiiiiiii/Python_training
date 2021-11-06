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
