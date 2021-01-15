import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from pythonlearn.tfmodels.rlrec.movie_env import *
import time

#ray.init()
from zoo.ray import RayContext
from zoo import init_spark_on_local

conf = {"spark.executor.memory": "20g", "spark.driver.memory": "20g"}
sc = init_spark_on_local(cores=8, conf=conf)
ray_ctx = RayContext(sc=sc, object_store_memory="2g")
ray_ctx.init()

config = ppo.DEFAULT_CONFIG.copy()
config["lr"] = 0.005
config["num_gpus"] = 0
config["num_workers"] = 8
#config["eager"] = False
config["env"]=MovieEnv

trainer = ppo.PPOTrainer(config=config)
# Can optionally call trainer.restore(path) to load a checkpoint.

t1 = time.time()
for i in range(1, 10000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 5 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
t2 = time.time()
print("total time", t2-t1)
# agent = ppo.PPOTrainer(config=config)
# agent.restore("/Users/guoqiong/ray_results/PPO_MovieEnv_2021-01-15_12-19-47e3t_a6pt/checkpoint_10/checkpoint-10")
# env = MovieEnv(config)
# # run until episode ends
# for i in range(100):
#     episode_reward = 0
#     done = False
#     obs = env.reset()
#     while not done:
#         action = agent.compute_action(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#         # if reward > 0:
#         print(reward, done, env._step )
#         print("episode_reward", episode_reward)
#

ray_ctx.stop()
