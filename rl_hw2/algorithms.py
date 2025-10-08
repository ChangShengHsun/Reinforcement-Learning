import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        state_return = np.zeros(self.grid_world.get_state_space())
        state_visited = np.zeros(self.grid_world.get_state_space())
        while self.episode_counter < self.max_episode:
            current_state = self.grid_world.reset()
            trajectory = []
            done_flag = False
            visited_in_episode = set()
            gain = 0
            while not done_flag:
                next_state, reward, done = self.collect_data()
                done_flag = done
                if current_state in visited_in_episode:
                    trajectory.append((current_state, reward, 0))
                else:
                    trajectory.append((current_state, reward, 1))
                    visited_in_episode.add(current_state)
                current_state = next_state
            for s in reversed(trajectory):
                gain = self.discount_factor * gain + s[1]
                if s[2] == 1:
                    state_return[s[0]] += gain
                    state_visited[s[0]] += 1
                    self.values[s[0]] = state_return[s[0]] / state_visited[s[0]]
                   

        return


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        while self.episode_counter < self.max_episode:
            current_state = self.grid_world.reset()
            done_flag = False
            while not done_flag:
                next_state, reward, done = self.collect_data()
                if done:
                    self.values[current_state] = self.values[current_state] + self.lr*(reward - self.values[current_state])
                else:
                    self.values[current_state] = self.values[current_state] + self.lr*(reward + self.discount_factor * self.values[next_state] - self.values[current_state])
                current_state = next_state
                done_flag = done


            continue


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        while self.episode_counter < self.max_episode:
            current_state = self.grid_world.reset()
            trajectory_state = []
            trajectory_state.append(current_state)
            trajectory_reward = []
            while True:
                next_state, reward, done = self.collect_data()
                trajectory_reward.append(reward)
                if done:
                    break
                trajectory_state.append(next_state)
                current_state = next_state

            T = len(trajectory_reward)
            for tau in range(T):
                gain = 0
                for i in range(tau, min(tau + self.n-1, T-1)+1):
                    gain += (self.discount_factor**(i - tau)) * trajectory_reward[i]
                if tau + self.n <= T-1:
                    gain += (self.discount_factor**self.n) * self.values[trajectory_state[tau + self.n]]
                self.values[trajectory_state[tau]] += self.lr * (gain - self.values[trajectory_state[tau]])

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        done_flag = False
        current_state = self.grid_world.reset()
        gain = 0
        max_steps = 100
        steps = 0
        while not done_flag and max_steps > steps:
            action_probs = self.policy[current_state]
            action = np.random.choice(self.action_space, p = action_probs)
            next_state, reward, done_flag = self.grid_world.step(action)
            state_trace.append(current_state)
            action_trace.append(action)
            reward_trace.append(reward)
            current_state = next_state
            steps += 1
        for s in reversed(list(zip(state_trace, action_trace, reward_trace))):
            gain = self.discount_factor * gain + s[2]
            self.q_values[s[0]][s[1]] += self.lr * (gain - self.q_values[s[0]][s[1]])
        return

        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        for s in range(self.state_space):
            best_action = self.q_values[s].argmax()
            for action in range(self.action_space):
                if action == best_action:
                    self.policy[s][action] = 1- self.epsilon + self.epsilon/self.action_space
                else:
                    self.policy[s][action] = self.epsilon/self.action_space
        return


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = []
        action_trace  = []
        reward_trace  = []
        while iter_episode < max_episode:
            self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()
            state_trace.clear()
            action_trace.clear()
            reward_trace.clear()
            iter_episode += 1


class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        if not is_done:
            self.q_values[s][a] += self.lr * (r + self.discount_factor * self.q_values[s2][a2] - self.q_values[s][a])
        else:
            self.q_values[s][a] += self.lr * (r - self.q_values[s][a])

        return is_done
    def epsilon_greedy(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        return int(self.q_values[state].argmax())


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        while iter_episode < max_episode:
            current_state = self.grid_world.reset()
            is_done = False
            max_steps = 10
            steps = 0
            while not is_done and steps < max_steps:
                prev_s = current_state
                prev_a = self.epsilon_greedy(current_state)
                next_state, prev_r, is_done = self.grid_world.step(prev_a)
                best_action = self.epsilon_greedy(next_state)
                is_done = self.policy_eval_improve(prev_s, prev_a, prev_r, next_state, best_action, is_done)
                steps += 1
                current_state = next_state
            iter_episode += 1
        for s in range(self.state_space):
            self.policy_index[s] = self.q_values[s].argmax()
            

class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        self.buffer.append((s,a,r,s2,d))

    def sample_batch(self) -> np.ndarray:
        batch_size = min(self.sample_batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        if is_done:
            self.q_values[s][a] += self.lr * (r - self.q_values[s][a])
        else:
            self.q_values[s][a] += self.lr * (r + self.discount_factor * self.q_values[s2].max() - self.q_values[s][a])
    def epsilon_greedy(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        return int(self.q_values[state].argmax())

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        transition_count = 0
        while iter_episode < max_episode:
            current_state = self.grid_world.reset()
            is_done  = False
            while not is_done:
                prev_s = current_state
                prev_a = self.epsilon_greedy(prev_s)
                next_state, prev_r, is_done = self.grid_world.step(prev_a)
                current_state = next_state
                self.add_buffer(prev_s, prev_a, prev_r, current_state, is_done)
                transition_count += 1
                batch = []
                if transition_count % self.update_frequency == 0:
                    batch  = self.sample_batch()
                    for s, a, r, s2, is_done_batch in batch:
                        self.policy_eval_improve(s,a,r,s2, is_done_batch)
            iter_episode += 1
        for s in range(self.state_space):
            self.policy_index[s] = self.q_values[s].argmax()
            