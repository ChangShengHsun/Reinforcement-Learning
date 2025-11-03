import warnings
import time
# --- 新增 imports ---
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=2, stride=1),  # -> (256, 3, 3)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1),# -> (512, 2, 2)
            nn.ReLU(inplace=True),
            nn.Flatten(),                                 # -> 512*2*2 = 2048
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, features_dim),
            nn.ReLU(inplace=True),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",
    "algorithm": DQN,
    # ✅ 改成 CNN policy
    "policy_network": "CnnPolicy",
    "save_path": "models/sample_model",
    "num_train_envs": 6,
    "epoch_num": 1000,
    "timesteps_per_epoch": 2048*6*2,
    "eval_episode_num": 10,

    "ent_coef": 0.025,
    # ✅ 專給 CnnPolicy 的參數：輸入已是 0/1，不做影像正規化
     "policy_kwargs": {
        "normalize_images": False,                   # 你的觀測已是 0/1 或小數，不需要 /255
        "features_extractor_class": CustomCNN,      # 使用上面自訂的 4x4 CNN
        "features_extractor_kwargs": {"features_dim": 128},
    },
}

def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API (VecEnv: 4-tuple)
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
        
    return avg_score, avg_highest

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_score = 0
    current_best_highest = 0

    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()

        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                verbose=2,
            ),
        )

        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time

        ### Evaluation
        eval_start = time.time()
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        eval_duration = time.time() - eval_start

        # Print training progress and speed
        if epoch % 10 == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
            print(f"{'='*60}")
            print(f"Training Speed:")
            print(f"   - Epoch time: {epoch_duration:.1f}s")
            print(f"   - Eval time: {eval_duration:.1f}s")
            print(f"   - Total time: {total_duration/60:.1f} min")
            print(f"Performance:")
            print(f"   - Avg Score: {avg_score:.1f}")
            print(f"   - Avg Highest Tile: {avg_highest:.1f}")

        wandb.log(
            {"avg_highest": avg_highest,
             "avg_score": avg_score,
            "epoch": epoch}
        )
        
        ### Save best model
        if current_best_score < avg_score or current_best_highest < avg_highest:
            print("Saving New Best Model")
            if current_best_score < avg_score:
                current_best_score = avg_score
                print(f"   - Previous best score: {current_best_score:.1f} → {avg_score:.1f}")
            elif current_best_highest < avg_highest:
                current_best_highest = avg_highest
                print(f"   - Previous best tile: {current_best_highest:.1f} → {avg_highest:.1f}")

            save_path = config["save_path"]
            model.save(f"{save_path}/best")
        print("-"*60)
            
    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")

if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    train_env = SubprocVecEnv([make_env for _ in range(my_config["num_train_envs"])])
    eval_env = DummyVecEnv([make_env])

    # Try to load previously trained 'best' model (best.zip). If loading fails, create a fresh model.
    save_dir = my_config["save_path"]
    best_model_name = f"{save_dir}/best"
    # try:
    #     print(f"Trying to load existing model '{best_model_name}.zip'...")
    #     # ⚠️ 若先前是用 MLP 存檔，這裡可能會失敗（架構不相容），下面 except 會改用新的 CNN 建立
    #     model = my_config["algorithm"].load(best_model_name, env=train_env, device='auto')
    #     print("Loaded existing model. Continuing training/evaluation with that model.")
    # except Exception as e:
    print(f"Could not load '{best_model_name}.zip' ; creating a new model.")
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=0,
        tensorboard_log=my_config["run_id"],
        # ✅ 把 policy_kwargs 傳入
        policy_kwargs=my_config.get("policy_kwargs", None)
    )

    train(eval_env, model, my_config)
