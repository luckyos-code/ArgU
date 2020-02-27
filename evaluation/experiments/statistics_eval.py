from statistics import mean, median, variance

names = ['R1_S', 'R2_S', 'R3_S', 'R1_M', 'R2_M', 'R3_M']
r1_s = [0, 0, 0.3, 0.1, -0.2, 0, 0, 0, 0, 0, 0, -0.1, -0.2]
r2_s = [-0.2, 0, -0.2, 0, 0.3, 0, 0, -0.1]
r3_s = [-0.1, -0.5, -0.2, -0.2, 0, -0.1, -0.1]

r1_m = [2.8, 1.8, 2.3, 3, 11, 27, 4.8, 4.8, 3.8, 0, 1.5, 19.7, 14.8]
r2_m = [2.4, 3, 4.1, 2.6, 1.4, 3.8, 5, 4]
r3_m = [4.6, 5.7, 2.6, 3.3, 1.8, 3.7, 3.5]

lists = [r1_s, r2_s, r3_s, r1_m, r2_m, r3_m]

for n, l in zip(names, lists):
    print(f'{n}: mean = {mean(l)}')
    print(f'{n}: median = {median(l)}')
    print(f'{n}: variance = {variance(l)}\n')
