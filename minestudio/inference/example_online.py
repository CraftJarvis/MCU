from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, CommandsCallback, JudgeResetCallback, FastResetCallback
from minestudio.models import VPTPolicy, load_vpt_policy

if __name__ == '__main__':
    
    policy = load_vpt_policy(
        model_path="pretrained/foundation-model-2x.model",
        weights_path="pretrained/rl-from-early-game-2x.weights"
    ).to("cuda")
    
    # env = MinecraftSim(
    #     obs_size=(128, 128), 
    #     preferred_spawn_biome="forest", 
    #     callbacks=[
    #         RecordCallback(record_path="./output", fps=30, frame_type="pov"),
    #         SpeedTestCallback(50),
    #     ]
    # )

    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="forest", 
        action_type = "agent",
        timestep_limit=1000,
        callbacks=[
            SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
            MaskActionsCallback(inventory=0), 
            RewardsCallback([{
                'event': 'mine_block', 
                'objects': ['oak_log', 'birch_log'], 
                'reward': 0.5, 
                'identity': 'chop_tree', 
                'max_reward_times': 30, 
            }]),
            CommandsCallback(commands=[
                '/give @p minecraft:iron_sword 1',
                '/give @p minecraft:diamond 64',
                '/effect @p 5 9999 255 true',
            ]),
            FastResetCallback(
                biomes=['forest'],
                random_tp_range=1000,
            ),
            JudgeResetCallback(600),
        ]
    )
    memory = None
    obs, info = env.reset()
    for i in range(600):
        for i in range(1000):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            print("reward:", reward)
            if reward > 0:
                import pdb; pdb.set_trace()
        obs, info = env.reset()
    env.close()