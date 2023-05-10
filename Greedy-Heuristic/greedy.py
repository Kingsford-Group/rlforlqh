import numpy as np
import time

n =10
k = 4

file1 = open("tests_10x10_[5,5,5,5]_sp.txt", "r+")
Lines = file1.readlines()

count = 0

sm = 0
def greedy(grid, goal):
    # print(grid)
    # print(goal)
    global count
    global sm

    head = np.zeros((k))
    done = False
    prev_x = 0
    prev_y = 0
    traveled = 0
    for r in range(100):
        if head.sum() == 0:
            grid_cand = []
            mn_dis = 2*n
            x, y = -1, -1
            for i in range(n):
                for j in range(n):
                    if grid[i][j].sum() > 0 and np.any(grid[i][j] > goal[i][j]):
                    # if grid[i][j].sum() > 0 and np.any(grid[i][j] != goal[i][j]):
                    #     grid_cand.append((i, j))
                        if abs(x-i) + abs(y-j) < mn_dis:
                            mn_dis = abs(x-i) + abs(y-j)
                            x, y = i, j

            # if len(grid_cand) == 0:
            #     continue
            # p = np.random.randint(0, len(grid_cand))
            # x, y = grid_cand[p]
            ratio_code = np.random.randint(0, 1)
            action_ratio = 1
            if ratio_code == 1 and grid[x][y].sum() > 1 \
                    and grid[x][y].sum() == grid[x][y].max():
                action_ratio = 1 / grid[x][y].sum()

            head = grid[x][y].copy() * action_ratio
            grid[x][y] -= head
            traveled += abs(x-prev_x) + abs(y-prev_y)
            prev_x = x
            prev_y = y
        else:
            goal_cand = []
            ratio_code = np.random.randint(0, 1)
            action_ratio = 1
            if ratio_code == 1 and head.sum() > 1 \
                    and head.sum() == head.max():
                action_ratio = 1 / head.sum()

            mn_dis = 2 * n
            x, y = -1, -1
            for i in range(n):
                for j in range(n):
                    if goal[i][j].sum() > 0 and np.all(head * action_ratio <= np.maximum(goal[i][j] - grid[i][j], 0)):
                    # if goal[i][j].sum() > 0 and np.any(grid[i][j] != goal[i][j]):
                    #     goal_cand.append((i, j))
                        if abs(x-i) + abs(y-j) < mn_dis:
                            mn_dis = abs(x-i) + abs(y-j)
                            x, y = i, j

            # if len(goal_cand) == 0:
            #     continue
            # p = np.random.randint(0, len(goal_cand))
            # x, y = goal_cand[p]
            grid[x][y] += head.copy() * action_ratio
            head *= (1 - action_ratio)
            done = np.all(goal == grid)
            traveled += abs(x - prev_x) + abs(y - prev_y)
            prev_x = x
            prev_y = y
            if done:
                break


    if done == False:
        count += 1
    else:
        sm += traveled
    print(count, done, r, traveled)


grid = np.zeros((n, n, k))
goal = np.zeros((n, n, k))

start = time.time()

# Strips the newline character
gg = False
for line in Lines:
    # print(line)
    if line[0] == "S":
        continue
    if line[0] == "G":
        gg = True
        continue
    if line[0] == "E" and not gg:
        continue
    if line[0] == "E" and gg:
        greedy(grid, goal)
        gg = False
        grid = np.zeros((n, n, k))
        goal = np.zeros((n, n, k))
        continue

    x, y, c = [int(x) for x in line.split()]
    # print("hello")
    if not gg:
        grid[x][y][c] += 1
    else:
        goal[x][y][c] += 1

end = time.time()

N = 1000
print(N-count)
print(sm/(N-count))

print("The time of execution of above program is :",
      (end-start) , "s")