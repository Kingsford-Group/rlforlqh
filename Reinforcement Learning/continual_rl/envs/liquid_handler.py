import gym
import numpy as np
from gym.spaces import Box, Discrete
import time


fixed_i = 0
sum_traveled = 0
class LiquidHandler(gym.Env):
    def __init__(self, grid_size=None, num_blocks=None, penalize_dist=True, full_reward=False, fixed_experiment=None):
        if grid_size is None:
            grid_size = [10, 10]

        if num_blocks is None:
            num_blocks = [0, 1]

        self._num_types = len(num_blocks)
        self._num_blocks = num_blocks
        self._penalize_dist = penalize_dist
        self._max_steps = 99
        self._fixed_experiment = fixed_experiment
        self._full_reward = full_reward
        self._cell_discretization = 2
        self._epsilon = 0.0001
        self._grid_size = grid_size
        self._total_traveled = 0
        global fixed_i
        self._fixed_experiment_i = fixed_i
        if fixed_experiment ==0:
            print(self._fixed_experiment_i)
        self._fixed_experiment_total = 100
        self._fixed_experiment_file = "tests_6x6_[3,3]_nsp.txt"
        self._fixed_grids = []
        self._fixed_goals = []

        self._grid = np.zeros((grid_size[0], grid_size[1], len(self._num_blocks)))
        self._goals = np.zeros(self._grid.shape)
        self._blocks_in_grasp = np.zeros(len(self._num_blocks,))
        self._current_step = 0
        self._current_arm_pos = np.array([0, 0])  # TODO?

        shape = [self._grid.shape[2] + self._goals.shape[2] + self._blocks_in_grasp.shape[0], self._grid.shape[0], self._grid.shape[1]]
        self.observation_space = Box(low=0, high=1.0, shape=shape, dtype=np.int32)
        self.action_space = Discrete(n=np.prod(self._grid.shape[:2])*self._cell_discretization)
        # if fixed_experiment == 0:
        #     self._init_set_experiments()

    def _populate_grid(self, grid, blocks_to_fill):
        for block_id, num_blocks in enumerate(blocks_to_fill):
            for filled_id in range(num_blocks):
                filled_pos = np.random.randint(0, grid.shape[:2])
                grid[filled_pos[0]][filled_pos[1]][block_id] += 1
                #print(f"Block {block_id} filled pos: {filled_pos}")

    def _init_set_experiments(self):
        file_goal = open(self._fixed_experiment_file, "r+")
        Lines = file_goal.readlines()
        grid = np.zeros((self._grid_size[0], self._grid_size[1], self._num_types))
        goal = np.zeros((self._grid_size[0], self._grid_size[1], self._num_types))
        gg = False
        for line in Lines:
            if line[0] == "S":
                continue
            if line[0] == "G":
                gg = True
                continue
            if line[0] == "E" and not gg:
                continue
            if line[0] == "E" and gg:
                gg = False
                self._fixed_grids.append(grid)
                self._fixed_goals.append(goal)

                grid = np.zeros((self._grid_size[0], self._grid_size[1], self._num_types))
                goal = np.zeros((self._grid_size[0], self._grid_size[1], self._num_types))
                continue

            x, y, c = [int(x) for x in line.split()]
            if not gg:
                grid[x][y][c] += 1
            else:
                goal[x][y][c] += 1
        print(len(self._fixed_grids), len(self._fixed_goals))
        # print(self._fixed_grids)

    def _get_fixed_grid(self, grid):
        grid = self._fixed_grids[self._fixed_experiment_i].copy()

    def _get_fixed_goal(self, goal):
        goal = self._fixed_goals[self._fixed_experiment_i].copy()
        self._fixed_experiment_i += 1
        if self._fixed_experiment_i == self._fixed_experiment_total:
            self._fixed_experiment_i = 0

    def _populate_grid_exp(self, grid):
        #   not mixing
        if self._fixed_experiment == 0:
            self._get_fixed_grid(grid)
        if self._fixed_experiment == 1:
            grid[0][0][0] = 1
            grid[0][0][1] = 1
            grid[1][0][0] = 1
            grid[2][0][1] = 1
        #   not mixing, move out test
        if self._fixed_experiment == 2:
            grid[0][-1][0] = 1
            grid[0][0][1] = 1
            grid[1][0][0] = 1
            grid[1][1][1] = 1
        #   check distance
        if self._fixed_experiment == 3:
            grid[0][0][0] = 1
            grid[-1][-1][0] = 1
        if self._fixed_experiment == 4:
            grid[0][0][0] = 1
            grid[-1][-1][1] = 1
        #   check partial
        if self._fixed_experiment == 5:
            grid[0][0][0] = 2
        if self._fixed_experiment == 6:
            grid[0][0][0] = 1
            grid[-1][-1][1] = 1

    def _populate_goal_exp(self, grid):
        if self._fixed_experiment == 0:
            self._get_fixed_goal(grid)
        if self._fixed_experiment == 1 or self._fixed_experiment == 2:
            grid[0][-1][0] = 1
            grid[0][-1][1] = 1
            grid[1][-1][0] = 1
            grid[2][-1][1] = 1
        if self._fixed_experiment == 3:
            grid[0][1][0] = 1
            grid[-1][-2][0] = 1
        if self._fixed_experiment == 4:
            grid[0][1][1] = 1
            grid[-1][-2][0] = 1
        if self._fixed_experiment == 5:
            grid[0][1][0] = 1
            grid[0][-1][0] = 1
        if self._fixed_experiment == 6:
            grid[0][0][1] = 1
            grid[-1][-1][0] = 1

    def _populate_grid_with_full_reward(self, goals):
        self._grid = np.zeros((self._grid_size[0], self._grid_size[1], len(self._num_blocks)))
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                blocks_to_fill = goals[i][j].copy()
                while blocks_to_fill.sum() > 0:
                    blocks = np.random.randint(0, blocks_to_fill+[1]*self._num_types)
                    while blocks.sum() == 0:
                        blocks = np.random.randint(0, blocks_to_fill + [1] * self._num_types)
                    filled_pos = np.random.randint(0, self._grid.shape[:2])
                    while np.minimum(blocks, goals[filled_pos[0]][filled_pos[1]]).sum() > 0 \
                            or self._grid[filled_pos[0]][filled_pos[1]].sum() != 0\
                            or goals[filled_pos[0]][filled_pos[1]].sum() != 0:
                        filled_pos = np.random.randint(0, self._grid.shape[:2])
                    self._grid[filled_pos[0]][filled_pos[1]] += blocks
                    blocks_to_fill -= blocks
        # print("goal:", goals, "\n grid: \n", self._grid, "\n\n")

    def _generate_observation(self):
        #return np.concatenate((self._grid.flatten(), self._goals.flatten()))
        tiled_blocks_in_grasp = np.tile(self._blocks_in_grasp, (self._grid.shape[0], self._grid.shape[1], 1))
        obs = np.concatenate((self._grid, self._goals, tiled_blocks_in_grasp), axis=-1)
        obs = obs.transpose((2, 0, 1))
        return obs

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        global fixed_i
        self._grid = np.zeros(self._grid.shape)
        self._goals = np.zeros(self._goals.shape)
        self._blocks_in_grasp = np.zeros(self._blocks_in_grasp.shape)
        self._current_step = 0
        self._current_arm_pos = np.array([0, 0])

        #print(f"Populating grid:")
        if self._fixed_experiment is None or self._fixed_experiment == 0:
            self._populate_grid(self._grid, self._num_blocks)
        else:
            self._populate_grid_exp(self._grid)

        #print(f"Populating goals:")
        if self._fixed_experiment is None or self._fixed_experiment == 0:
            if self._full_reward:
                self._populate_grid(self._goals, self._num_blocks)
                self._populate_grid_with_full_reward(self._goals)
                if self._fixed_experiment == 0:
                    fixed_i += 1
                    print("LH", fixed_i)
            else:
                self._populate_grid(self._goals, self._num_blocks)
        else:
            self._populate_goal_exp(self._goals)

        return self._generate_observation()

    def step(self, action):
        action_ratio_code = (action % self._cell_discretization)
        action_ratio = 1
        action //= self._cell_discretization
        action_x = action // self._grid.shape[1]
        action_y = action % self._grid.shape[0]
        new_arm_pos = np.array([action_x, action_y])

        dist_traveled = (abs(new_arm_pos - self._current_arm_pos)).sum()
        self._total_traveled += dist_traveled
        self._current_arm_pos = new_arm_pos

        # If there are no blocks in the grasp, it'll perform a pick action
        if self._blocks_in_grasp.sum() == 0:

            if action_ratio_code == 1 and self._grid[action_x][action_y].sum() > 1\
                    and self._grid[action_x][action_y].sum() == self._grid[action_x][action_y].max():
                action_ratio = 1 / self._grid[action_x][action_y].sum()
                # action_ratio = 1
                # print(action_ratio, self._grid[action_x][action_y].sum())

            # Penalize if we're picking up a goal that was already completed
            completed_goals = np.minimum(self._goals[action_x][action_y], self._grid[action_x][action_y])
            new_goals = np.minimum(self._goals[action_x][action_y], self._grid[action_x][action_y]*(1-action_ratio))
            removed_goals = completed_goals - new_goals
            unnecessary_blocks = self._grid[action_x][action_y]*action_ratio - removed_goals

            # Reward for any non-goal blocks picked up (for net-zero consistency), and penalize for any goal blocks picked
            reward = unnecessary_blocks.sum() - removed_goals.sum()

            # if reward < 0:
            #     print("Case 1:", action_ratio, self._goals[action_x][action_y], self._grid[action_x][action_y], reward, unnecessary_blocks, completed_goals)


            # Pick blocks
            #print(f"Picking from [{action_x}][{action_y}]")
            self._blocks_in_grasp = self._grid[action_x][action_y].copy()*action_ratio
            self._grid[action_x][action_y] *= (1-action_ratio)
            #print(f"Blocks held: {self._blocks_in_grasp}")

            done = False
        else:
            if action_ratio_code == 1 and self._blocks_in_grasp.sum() > 1 \
                    and self._blocks_in_grasp.sum() == self._blocks_in_grasp.max():
                action_ratio = 1 / self._blocks_in_grasp.sum()
                # action_ratio = 1


            # Otherwise, we're in a place state
            #print(f"Placing {self._blocks_in_grasp} into [{action_x}][{action_y}]")

            # How many blocks of each type we have left to get
            goal_blocks_left = np.clip(self._goals[action_x][action_y] - self._grid[action_x][action_y], a_min=0, a_max=None)

            # If we're placing more blocks than desired, only count the number desired
            # If we're placing fewer, only count the blocks placed
            goal_blocks_placed = np.minimum(self._blocks_in_grasp*action_ratio, goal_blocks_left)
            unnecessary_blocks = self._blocks_in_grasp*action_ratio - goal_blocks_placed
            #print(f"Reward from plac: {reward}")

            # Reward for the goal blocks, but penalize for placing extras unnecessarily
            reward = goal_blocks_placed.sum() - unnecessary_blocks.sum()

            # if reward < 0:
            #     print("Case 2:", action_ratio, self._goals[action_x][action_y], self._grid[action_x][action_y], self._blocks_in_grasp, goal_blocks_left, reward, goal_blocks_placed, unnecessary_blocks)

            # Place the blocks
            self._grid[action_x][action_y] += self._blocks_in_grasp.copy()*action_ratio
            self._blocks_in_grasp *= (1-action_ratio)

            # done = np.all(abs(self._goals - self._grid) <= self._epsilon)
            done = np.all(self._goals == self._grid)

            # print("\n\n")
            # print(self._goals, self._grid, done)
            # print("\n\n")
        obs = self._generate_observation()
        self._current_step += 1
        done = done or self._current_step > self._max_steps

        if self._fixed_experiment == 0 and done:
        # if done:
            print(self._fixed_experiment_i, np.all(self._goals == self._grid), self._total_traveled)
            print(np.all(self._goals == self._grid), self._total_traveled, time.time(), file=open('test_time_10x10_[5,5,5,5].txt', 'a'))


        # if self._current_step > 10:
        #     print("goal:", self._goals, "\n grid: \n", self._grid, "\n\n")

        if self._penalize_dist:
            max_dist = self._grid_size[0] + self._grid_size[1]
            reward -= 0.2 * dist_traveled / max_dist

        return obs, reward, done, {}


if __name__ == "__main__":
    env = LiquidHandler([2, 2])
    obs = env.reset()

    test = LiquidHandler(grid_size=[5, 5], num_blocks=[12], penalize_dist=False, full_reward=True)
    obs2 = test.reset()
    print("done")

    done = False

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
