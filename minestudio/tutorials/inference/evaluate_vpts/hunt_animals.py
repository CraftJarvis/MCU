'''
Date: 2024-12-13 14:31:12

LastEditTime: 2024-12-13 15:23:35
FilePath: /MineStudio/minestudio/tutorials/inference/evaluate_hunt_vpt.py
'''
import ray
from rich import print
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter

from functools import partial
from minestudio.models import load_vpt_policy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    SpeedTestCallback, 
    SummonMobsCallback, 
    FastResetCallback,
    CommandsCallback
)

if __name__ == '__main__':
    ray.init()
    env_generator = partial(
        MinecraftSim, 
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        callbacks=[
            SpeedTestCallback(50), 
            SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=1000,
            ), 
            CommandsCallback(commands=[
                '/give @p minecraft:iron_sword 1',
            ]), 
        ]
    )
    agent_generator = partial(
        load_vpt_policy,
        model_path="pretrained/foundation-model-2x.model",
        # weights_path="pretrained/foundation-model-1x.weights"
        weights_path="//minestudio/save/2024-12-13/23-01-45/weights/weight-epoch=2-step=1000.ckpt", 
    )
    worker_kwargs = dict(
        env_generator=env_generator, 
        agent_generator=agent_generator,
        num_max_steps=600,
        num_episodes=2,
        tmpdir="./output",
        image_media="h264",
    )
    pipeline = EpisodePipeline(
        episode_generator=MineGenerator(
            num_workers=2,
            num_gpus=0.25,
            max_restarts=3,
            **worker_kwargs, 
        ), 
        episode_filter=InfoBaseFilter(
            key="kill_entity",
            val="cow",
            num=1,
        ),
    )
    summary = pipeline.run()
    print(summary)