class AsyncDynamicProgramming(DynamicProgramming):
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
    
    def inplace_DP(self):
        new_state_value = 0
        max_delta = 0
        for state in range(self.grid_world.get_state_space()):
            new_state_value = self.get_state_value(state)
            if(abs(new_state_value - self.values[state]) > max_delta):
                max_delta = abs(new_state_value - self.values[state])
            self.values[state] = new_state_value
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
        # TODO: Implement the async dynamic programming algorithm until convergence
        delta = 2*self.threshold
        while delta > self.threshold:
            delta = self.inplace_DP()
        self.policy_improvement()
