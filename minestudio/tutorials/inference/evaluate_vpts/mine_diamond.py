'''
Date: 2024-11-25 08:11:33

LastEditTime: 2024-12-15 14:17:07
FilePath: /MineStudio/minestudio/tutorials/inference/evaluate_vpts/mine_diamond.py
'''
import ray
from rich import print
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter

from functools import partial
from minestudio.models import load_vpt_policy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import SpeedTestCallback

if __name__ == '__main__':
    ray.init()
    env_generator = partial(
        MinecraftSim, 
        obs_size=(128, 128), 
        preferred_spawn_biome="forest", 
        callbacks=[
            SpeedTestCallback(50), 
        ],
    )
    agent_generator = partial(
        load_vpt_policy,
        model_path="pretrained/foundation-model-2x.model",
        weights_path="pretrained/rl-from-early-game-2x.weights"
    )
    worker_kwargs = dict(
        env_generator=env_generator, 
        agent_generator=agent_generator,
        num_max_steps=12000,
        num_episodes=2,
        tmpdir="./output",
        image_media="h264",
    )
    pipeline = EpisodePipeline(
        episode_generator=MineGenerator(
            num_workers=8,
            num_gpus=0.25,
            max_restarts=3,
            **worker_kwargs, 
        ), 
        episode_filter=InfoBaseFilter(
            key="mine_block",
            val="diamond_ore",
            num=1,
        ),
    )
    summary = pipeline.run()
    print(summary)