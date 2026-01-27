from ai_runner import ppo_train_unsupervised

if __name__ == "__main__":
    ppo_train_unsupervised(host="localhost:50051", num_envs=10)