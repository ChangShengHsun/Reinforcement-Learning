import numpy as np
import random
import heapq
from gridworld import GridWorld


class DynamicProgramming:
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        next_state, reward, done = self.grid_world.step(state, action)
        if done:
            return reward
        else:
            return reward + self.discount_factor*self.values[next_state]
    


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        state_value = 0.0
        for i in range(self.grid_world.get_action_space()):
            state_value += self.policy[state][i]*self.get_q_value(state, i)
            
        return state_value
        # TODO: Get the value for a state by calculating the q-values

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        new_state_values = self.values.copy()
        max_delta = 0
        for state in range(self.grid_world.get_state_space()):
            new_state_values[state] = self.get_state_value(state)
            if(abs(new_state_values[state] - self.values[state]) > max_delta):
                max_delta = abs(new_state_values[state] - self.values[state])
        self.values = new_state_values.copy()
        return max_delta

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        delta = 2*self.threshold
        while delta > self.threshold:
            delta = self.evaluate()

class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        state_value = self.get_q_value(state, self.policy[state]) 
        return state_value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        new_state_values = self.values.copy()
        max_delta = 0
        for state in range(self.grid_world.get_state_space()):
            new_state_values[state] = self.get_state_value(state)
            if(abs(new_state_values[state] - self.values[state]) > max_delta):
                    max_delta = abs(new_state_values[state] - self.values[state])
        self.values = new_state_values.copy()
        return max_delta
        

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        stable = True
        for state in range(self.grid_world.get_state_space()):
            q_pi_value = []
            for a in range(self.grid_world.get_action_space()):
                q_pi_value.append(self.get_q_value(state,a))
            best_action = q_pi_value.index(max(q_pi_value))
            if(best_action != self.policy[state]):
                stable = False
            self.policy[state] = best_action
        return stable


    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        stable = False
        while not stable:
            delta = 2*self.threshold
            while delta > self.threshold:
                delta = self.policy_evaluation()
            # self.policy_evaluation()
            stable =self.policy_improvement()


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        q_pi_value = []
        for a in range(self.grid_world.get_action_space()):
            q_pi_value.append(self.get_q_value(state,a))
        new_value = max(q_pi_value)
        return new_value
    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        new_state_values = self.values.copy()
        max_delta = 0
        for state in range(self.grid_world.get_state_space()):
            new_state_values[state] = self.get_state_value(state)
            if(abs(new_state_values[state] - self.values[state]) > max_delta):
                max_delta = abs(new_state_values[state] - self.values[state])
        self.values = new_state_values.copy()
        return max_delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            q_pi_value = []
            for a in range(self.grid_world.get_action_space()):
                q_pi_value.append(self.get_q_value(state,a))
            best_action = q_pi_value.index(max(q_pi_value))
            self.policy[state] = best_action
        

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        delta = 2*self.threshold
        while delta > self.threshold:
            delta = self.policy_evaluation()
        self.policy_improvement()


# Optimized prioritized sweeping
class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.predecessors = [[] for _ in range(grid_world.get_state_space())]
        # Initialize legal_actions for each state with all possible actions
        self.legal_actions = []
        for state in range(grid_world.get_state_space()):
            self.legal_actions.append(list(range(grid_world.get_action_space())))

        super().__init__(grid_world, discount_factor)
        self.values = self.values - 1e18

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # Only calculate from legal actions and remove wall-hitting actions
        q_pi_value = []
        actions_to_remove = []
        
        for a in self.legal_actions[state]:
            next_state, reward, done = self.grid_world.step(state, a)
            if next_state == state and not done:  # Wall-hitting action
                actions_to_remove.append(a)
            else:
                if done:
                    q_pi_value.append(reward)
                else:
                    q_pi_value.append(reward + self.discount_factor*self.values[next_state])
        
        # Remove wall-hitting actions from legal_actions
        for a in actions_to_remove:
            if a in self.legal_actions[state]:
                self.legal_actions[state].remove(a)
        
        # If no legal actions remain, return current value
        if not q_pi_value:
            return self.values[state]
            
        new_value = max(q_pi_value)
        return new_value


    def Prioritized_sweeping(self):
        nS = self.grid_world.get_state_space()
        nA = self.grid_world.get_action_space()
        eps = self.threshold

        residual = np.zeros(nS)
        new_values = np.zeros(nS)

        for state in range(nS):
            new_val = float('-inf')
            for a in range(nA):
                next_state, reward, done = self.grid_world.step(state, a)
                if done:
                    q_val = reward
                else:
                    q_val =  reward + self.discount_factor*self.values[next_state]
                if q_val > new_val:
                    new_val = q_val

                if next_state == state:
                    continue
                if state not in self.predecessors[next_state]:
                    self.predecessors[next_state].append(state)

            residual[state] = abs(new_val - self.values[state])
            new_values[state] = new_val

        while True:
            s = np.argmax(residual)
            if residual[s] < eps:
                break
            curr_new = self.get_state_value(s)
            err = abs(curr_new - self.values[s])
            if err < eps: 
                residual[s] = 0; continue
            self.values[s] = curr_new
            residual[s] = 0

            for pred in self.predecessors[s]:
                pred_new_value = self.get_state_value(pred)
                pred_residual = abs(pred_new_value - self.values[pred])
                if pred_residual > eps:
                    new_values[pred] = pred_new_value
                    residual[pred] = pred_residual
                    
                    
    
    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            # Only consider legal actions (non-wall-hitting actions)
            if not self.legal_actions[state]:
                # If no legal actions, keep current policy
                continue
                
            q_pi_value = []
            action_indices = []
            for a in self.legal_actions[state]:
                q_pi_value.append(self.get_q_value(state, a))
                action_indices.append(a)
            
            if q_pi_value:  # Only update if we have legal actions
                best_idx = q_pi_value.index(max(q_pi_value))
                best_action = action_indices[best_idx]
                self.policy[state] = best_action
    
    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        
        self.Prioritized_sweeping()
        self.policy_improvement()
