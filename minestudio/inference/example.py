'''
Date: 2024-11-14 19:42:09

LastEditTime: 2024-12-15 13:36:22
FilePath: /MineStudio/minestudio/inference/example.py
'''
import torch
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, CommandsCallback, FastResetCallback, JudgeResetCallback
from minestudio.models import VPTPolicy, load_vpt_policy
import os
import subprocess

def check_and_kill_process():
    # 获取系统内存使用率
    memory_usage = psutil.virtual_memory().percent
    print(f"System memory utilization: {memory_usage:.2f}%")
    
    if memory_usage > 90:  # 超过90%
        current_pid = os.getpid()  # 获取当前进程的PID
        print(f"Memory utilization exceeded 90%. Terminating process with PID {current_pid}...")
        os.kill(current_pid, 9)  # 强制杀掉当前进程


if __name__ == '__main__':
    
    policy = load_vpt_policy(
        model_path="pretrained/foundation-model-2x.model",
        weights_path="pretrained/foundation-model-2x.weights", 
    ).to("cuda")
    policy.eval()
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SummonMobsCallback, 
        MaskActionsCallback, 
        RewardsCallback, 
        CommandsCallback, 
        JudgeResetCallback,
        FastResetCallback,
        GateRewardsCallback,
        VoxelsCallback,
    )
    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        callbacks=[
            MaskActionsCallback(inventory = 0), 
            GateRewardsCallback(),
            CommandsCallback(commands=[
                '/give @p minecraft:diamond_pickaxe 1',
                '/replaceitem entity @s weapon.offhand minecraft:obsidian 64'
            ]),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=1000,
            ),
            JudgeResetCallback(2400),
            VoxelsCallback()
        ]
    )
    
    memory = None
    obs, info = env.reset()
    reward_sum = 0
    with open("output/resulttt.txt", "a") as f:
        f.write("------------------------\n")
    for j in range(100):
        for i in range(600):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum+=reward
        env.reset()
        print("Resetting the environment\n") 
        for i in range(600):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum+=reward
        env.reset()
        print("reward_sum: ", reward_sum)
