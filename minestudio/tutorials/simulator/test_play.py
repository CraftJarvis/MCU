import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback, RecordCallback, PointCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, CommandsCallback, TaskCallback, JudgeResetCallback, FastResetCallback
)
from minestudio.simulator.utils.gui import RecordDrawCall, CommandModeDrawCall, SegmentDrawCall
from functools import partial
from minestudio.models import load_vpt_policy, load_rocket_policy
if __name__ == '__main__':
    agent_generator = partial(
        load_rocket_policy,
        ckpt_path="/home/zhwang/Desktop//jarvisbase/pretrained/rocket_12-01.ckpt",
    )
    sim = MinecraftSim(
            obs_size=(128, 128), 
            preferred_spawn_biome="plains", 
            action_type = "agent",
            timestep_limit=1000,
            callbacks=[
                #SpeedTestCallback(50), 
                SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
                MaskActionsCallback(inventory=0), 
                RewardsCallback([{
                    'event': 'kill_entity', 
                    'objects': ['cow', 'sheep'], 
                    'reward': 5.0, 
                    'identity': 'kill sheep or cow', 
                    'max_reward_times': 5, 
                }]),
                CommandsCallback(commands=[
                    '/give @p minecraft:iron_sword 1',
                    '/give @p minecraft:diamond 64',
                ]), 
                FastResetCallback(
                    biomes=['mountains'],
                    random_tp_range=1000,
                ), 
                JudgeResetCallback(600),
            ]
        )
    sim.reset()
    for i in range(100):
        for j in range(100):
            obs, reward, terminated, truncated, info = sim.step(sim.noop_action())
        obs, info = sim.reset()
        print("info['kill_entity']: ", info['kill_entity'])
    terminated = False

    # while not terminated:
    #     action = None
    #     obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()