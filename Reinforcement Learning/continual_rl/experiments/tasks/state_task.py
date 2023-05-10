import torch
import gym
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
from continual_rl.utils.env_wrappers import FrameStack


class StateToPyTorch(gym.ObservationWrapper):
    # TODO (Issue 50): If used after LazyFrames, seems to negate the point of LazyFrames
    # As in, LazyFrames provides no benefit?
    # For now switching this to return a Tensor and calling it *before* FrameStack...

    def __init__(self, env, dict_space_key=None):
        super().__init__(env)
        self._key = dict_space_key

    def observation(self, observation):
        state_observation = observation if self._key is None else observation[self._key]
        processed_observation = torch.as_tensor(state_observation)

        if self._key is not None:
            observation[self._key] = processed_observation
        else:
            observation = processed_observation

        return observation


class StatePreprocessor(PreprocessorBase):
    def __init__(self, time_batch_size, env_spec):
        self.env_spec = self._wrap_env(env_spec, time_batch_size)

        dummy_env, _ = Utils.make_env(self.env_spec)
        observation_space = dummy_env.observation_space
        dummy_env.close()
        del dummy_env

        super().__init__(observation_space)

    def _wrap_env(self, env_spec, time_batch_size):
        # Leverage the existing env wrappers for simplicity
        frame_stacked_env_spec = lambda: FrameStack(StateToPyTorch(Utils.make_env(env_spec)[0]), time_batch_size)
        return frame_stacked_env_spec

    def preprocess(self, batched_env_states):
        """
        The preprocessed image will have values in range [0, 255] and shape [batch, time, channels, width, height].
        Handled as a batch for speed.
        """
        processed_state = torch.stack([state.to_tensor() for state in batched_env_states])
        return processed_state

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        # [batch, time, channels, height, width]
        # TODO: move this into the environment

        stacked_observations = torch.stack(episode_observations)

        # Assuming the observation channels correspond to [N grid states, N goal states, N blocks-in-hand states]
        num_block_types = episode_observations[0].shape[0] // 3

        # Here we're converting every grid cell into [5, 2 + N] pixels, where each block type's presence is handled
        # by a pixel, and the columns correspond to the order specified above. The extra 2 in each dim is whitespace
        # for clarity
        split_columns = stacked_observations.view((stacked_observations.shape[0], 3, num_block_types, *stacked_observations.shape[2:]))

        # TODO: determining number of blocks by (grid + held in hand). Could use goal, assuming a block always has a home, but not making that assumption here
        # The hand contents are replicated for every cell, so we just grab the first
        num_blocks_per_type = (split_columns[:, 0].sum(dim=2).sum(dim=2) + split_columns[:, 2, :, 0, 0]).max(dim=0)[0]

        # Generate the values we'll use for the colors -- divide by the num_blocks so we get darker colors the more blocks there are in-hand
        color_spec = 1 - torch.tensor([[(block_count - 1 - i)/(block_count - 1), 1, i/(block_count-1)] for i, block_count in enumerate(num_blocks_per_type)])/num_blocks_per_type.unsqueeze(1)
        obs_colors = color_spec.permute(1, 0).unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(3) * split_columns.unsqueeze(1).permute(0, 1, 3, 2, 4, 5)
        obs_colors_padded = torch.nn.functional.pad(obs_colors, (0, 0, 0, 0, 1, 1, 1, 1), value=1)
        obs_colors_combined = obs_colors_padded.permute(0, 1, 4, 2, 5, 3).flatten(2, 3).flatten(3, 4)
        return obs_colors_combined.unsqueeze(0)  # Include batch


class StateTask(TaskBase):
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode,
                 continual_eval=True, continual_eval_num_returns=10):
        preprocessor = StatePreprocessor(time_batch_size, env_spec)

        dummy_env, _ = Utils.make_env(preprocessor.env_spec)
        action_space = dummy_env.action_space
        dummy_env.close()
        del dummy_env

        super().__init__(task_id, action_space_id, preprocessor, preprocessor.env_spec, preprocessor.observation_space,
                         action_space, num_timesteps, eval_mode, continual_eval=continual_eval,
                         continual_eval_num_returns=continual_eval_num_returns)
