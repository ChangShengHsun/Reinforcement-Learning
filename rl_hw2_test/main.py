import random
from matplotlib.table import table
import numpy as np
import json
import wandb


from algorithms import (
    MonteCarloPrediction,
    TDPrediction,
    NstepTDPrediction,
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld
# wandb.init(project="rl_hw2", name="prediction_variance")
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 2-1
STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
POLICY          = None
MAX_EPISODE     = 300
LEARNING_RATE   = 0.01
NUM_STEP        = 3
# 2-2
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500

def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_MC_prediction(grid_world: GridWorld,seed):
    print(f"Run MC prediction. Seed:{seed}")
    prediction = MonteCarloPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"Monte Carlo Prediction",
        show=False,
        filename=f"MC_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_TD_prediction(grid_world: GridWorld, seed):
    print(f"Run TD(0) prediction. Seed:{seed}")
    prediction = TDPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        learning_rate=LEARNING_RATE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"TD(0) Prediction",
        show=False,
        filename=f"TD0_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_NstepTD_prediction(grid_world: GridWorld,seed):
    print(f"Run N-step TD prediction. Seed:{seed}")
    prediction = NstepTDPrediction(
        grid_world,
        learning_rate=LEARNING_RATE,
        num_step=NUM_STEP,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed=seed,
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"N-step TD Prediction",
        show=False,
        filename=f"NstepTD_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"MC Policy Iteration",
        show=False,
        filename=f"MC_policy_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

def run_SARSA(grid_world: GridWorld, iter_num: int):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"SARSA",
        show=False,
        filename=f"SARSA_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

if __name__ == "__main__":
#     grid_world = init_grid_world("maze.txt",INIT_POS)
#     # 2-1
#     monte_carlo_values = []
#     td_values = []
#     ground_truth_values = np.load("prediction_GT.npy")
#     for seed in range(50):
#         monte_carlo_values.append(run_MC_prediction(grid_world,seed))
#         td_values.append(run_TD_prediction(grid_world,seed))
#         #calculate average
#         mc_avg_values = np.mean(monte_carlo_values, axis=0)
#         td_avg_values = np.mean(td_values, axis=0)
#         #calculate variance
#         mc_var_values = np.var(monte_carlo_values, axis=0)
#         td_var_values = np.var(td_values, axis=0)
#         #calculate bias
#         mc_bias_values = (mc_avg_values - ground_truth_values)
#         td_bias_values = (td_avg_values - ground_truth_values)
#     var_table  = wandb.Table(columns=["state", "series", "value"])
#     bias_table = wandb.Table(columns=["state", "series", "value"])
#     num_states = len(ground_truth_values)
#     for i in range(num_states):
#         # Variance 長表
#         var_table.add_data(int(i), "MC", float(mc_var_values[i]))
#         var_table.add_data(int(i), "TD", float(td_var_values[i]))
#         # Bias^2 長表
#         bias_table.add_data(int(i), "MC", float(mc_bias_values[i]))
#         bias_table.add_data(int(i), "TD", float(td_bias_values[i]))
#     wandb.log({
#     "Variance vs State": wandb.plot.line(
#         var_table, x="state", y="value", stroke="series", title="Variance vs State"
#     ),
#     "Bias vs State": wandb.plot.line(
#         bias_table, x="state", y="value", stroke="series", title="Bias vs State"
#     ),
# })


    # 2-2
    # 2-2 控制實驗 + wandb logging
    grid_world = init_grid_world("maze.txt")
    methods = ["sarsa", "qlearning"]
    epsilons = [0.1, 0.2, 0.3, 0.4]

    for method in methods:
        for epsilon in epsilons:
            run_name = f"{method}_epsilon={epsilon}"
            wandb.init(project="rl_hw2_control", name=run_name, reinit=True)
            EPSILON = epsilon

            if method == "montecarlo":
                print(f"\n===== Running Monte Carlo with epsilon={epsilon} =====")
                policy_iteration = MonteCarloPolicyIteration(
                    grid_world,
                    discount_factor=DISCOUNT_FACTOR,
                    learning_rate=LEARNING_RATE,
                    epsilon=EPSILON,
                )
                trace = policy_iteration.run(max_episode=512000)

            elif method == "sarsa":
                print(f"\n===== Running SARSA with epsilon={epsilon} =====")
                policy_iteration = SARSA(
                    grid_world,
                    discount_factor=DISCOUNT_FACTOR,
                    learning_rate=LEARNING_RATE,
                    epsilon=EPSILON,
                )
                trace = policy_iteration.run(max_episode=512000)

            elif method == "qlearning":
                print(f"\n===== Running Q-Learning with epsilon={epsilon} =====")
                policy_iteration = Q_Learning(
                    grid_world,
                    discount_factor=DISCOUNT_FACTOR,
                    learning_rate=LEARNING_RATE,
                    epsilon=EPSILON,
                    buffer_size=BUFFER_SIZE,
                    update_frequency=UPDATE_FREQUENCY,
                    sample_batch_size=SAMPLE_BATCH_SIZE,
                )
                trace = policy_iteration.run(max_episode=50000)

            # --- wandb log ---
            # 每個 episode 都 log reward / loss
            for ep, reward, loss in trace:
                wandb.log({
                    "episode": ep,
                    "average_reward": reward,
                    "average_loss": loss,
                    "epsilon": epsilon,
                    "method": method
                })

            # 結束當前 run
            wandb.finish()
            grid_world.reset()
            print(f"✅ Finished {run_name}")

        

    

    # run_MC_policy_iteration(grid_world, 512000)
    # run_SARSA(grid_world, 512000)
    # run_Q_Learning(grid_world, 50000)