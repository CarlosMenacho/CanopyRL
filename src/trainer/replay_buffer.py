import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, img_shape: tuple[int, int, int]) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.pixels = np.zeros((capacity, *img_shape), dtype=np.uint8)
        self.next_pixels = np.zeros((capacity, *img_shape), dtype=np.uint8)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: dict, action, reward, next_obs: dict, done) -> None:
        self.state[self.ptr] = obs["state"]
        self.pixels[self.ptr] = obs["pixels"]
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_obs["state"]
        self.next_pixels[self.ptr] = next_obs["pixels"]
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": self.state[idx],
            "pixels": self.pixels[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_state": self.next_state[idx],
            "next_pixels": self.next_pixels[idx],
            "done": self.done[idx],
        }

    def __len__(self) -> int:
        return self.size
