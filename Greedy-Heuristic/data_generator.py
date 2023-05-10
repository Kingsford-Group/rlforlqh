import numpy as np

n = 8
blocks = [15, 15]
k = len(blocks)


def _populate_grid(blocks_to_fill, grid, printing=True):
    for block_id, num_blocks in enumerate(blocks_to_fill):
        for filled_id in range(num_blocks):
            filled_pos = np.random.randint(0, [n, n])
            grid[filled_pos[0]][filled_pos[1]][block_id] += 1
            # print(f"Block {block_id} filled pos: {filled_pos}")
            if printing:
                print(filled_pos[0], filled_pos[1], block_id)
    if printing:
        print("END")
    return grid


def _populate_goal_with_full_reward(goals, printing, separate_plates):
    grid = np.zeros((n, n, k))
    for i in range(n):
        for j in range(n):
            blocks_to_fill = goals[i][j].copy()
            while blocks_to_fill.sum() > 0:
                blocks = np.random.randint(0, blocks_to_fill+[1]*k)
                while blocks.sum() == 0:
                    blocks = np.random.randint(0, blocks_to_fill + [1] * k)
                filled_pos = np.random.randint(0, grid.shape[:2])
                while np.minimum(blocks, goals[filled_pos[0]][filled_pos[1]]).sum() > 0 \
                        or grid[filled_pos[0]][filled_pos[1]].sum() != 0\
                        or (separate_plates and goals[filled_pos[0]][filled_pos[1]].sum() != 0):
                    filled_pos = np.random.randint(0, grid.shape[:2])
                grid[filled_pos[0]][filled_pos[1]] += blocks
                for b in range(len(blocks)):
                    for _ in range(blocks[b]):
                        if printing:
                            print(filled_pos[0], filled_pos[1], b)
                blocks_to_fill -= blocks
    if printing:
        print("END")
    return grid


def _populate_grid_exp(_fixed_experiment):
    #   not mixing
    if _fixed_experiment == 1:
        print(0, 0, 0)
        print(0, 0, 1)
        print(1, 0, 0)
        print(1, 1, 1)
    #   not mixing, move out test
    if _fixed_experiment == 2:
        print(0, n-1, 0)
        print(0, 0, 1)
        print(1, 0, 0)
        print(1, 1, 1)
    #   check distance
    if _fixed_experiment == 3:
        print(0, 0, 0)
        print(n-2, n-2, 0)
    if _fixed_experiment == 4:
        print(0, 0, 0)
        print(n-2, n-2, 1)
    if _fixed_experiment == 5:
        print(0, 0, 0)
        print(0, 0, 0)
    print("END")


def _populate_goal_exp(_fixed_experiment):
    if _fixed_experiment == 1 or _fixed_experiment == 2:
        print(0, n-2, 0)
        print(0, n-2, 1)
        print(1, n-2, 0)
        print(1, n-1, 1)
    if _fixed_experiment == 3:
        print(0, 1, 0)
        print(n-2, n-1, 0)
    if _fixed_experiment == 4:
        print(0, 1, 1)
        print(n-2, n-1, 0)
    if _fixed_experiment == 5:
        print(n-1, n-1, 0)
        print(n-1, n-2, 0)
    print("END")


if __name__ == '__main__':
    grid = np.zeros((n, n, k))
    np.random.seed(0)
    i = 0
    for _ in range(0):
        print(f"STARTING STATE {i}")
        _populate_grid(blocks, grid, True)
        print(f"GOAL STATE {i}")
        _populate_grid(blocks, grid, True)
        i += 1
    for _ in range(1000):
        goal = np.zeros((n, n, k))
        print(f"STARTING STATE {i}")
        _populate_grid(blocks, goal, False)
        _populate_goal_with_full_reward(goal, True, True)
        print(f"GOAL STATE {i}")
        for x in range(n):
            for y in range(n):
                for b in range(k):
                    for _ in range(int(goal[x][y][b])):
                        print(x, y, b)
        print("END")
        i += 1
    for _ in range(0):
        goal = np.zeros((n, n, k))
        print(f"STARTING STATE {i}")
        _populate_grid(blocks, goal, False)
        _populate_goal_with_full_reward(goal, True, False)
        print(f"GOAL STATE {i}")
        for x in range(n):
            for y in range(n):
                for b in range(k):
                    for _ in range(int(goal[x][y][b])):
                        print(x, y, b)
        print("END")
        i += 1

