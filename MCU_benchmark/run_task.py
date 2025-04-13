import argparse
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, JudgeResetCallback, FastResetCallback
from minestudio.models import VPTPolicy, load_vpt_policy
import torch
import os
# from minestudio.online.run.commands import CommandsCallback
from minestudio.models.steve_one import SteveOnePolicy, load_steve_one_policy
import numpy as np

def extract_info(yaml_content, filename):
    lines = yaml_content.splitlines()
    commands = []
    text = ''

    for line in lines:
        if line.startswith('-'):
            command = line.strip('- ').strip()
            commands.append(command)
        elif line.startswith('text:'):
            text = line.strip('text: ').strip()

    filename = filename[:-5].replace('_', ' ')

    print("File:", filename)
    print("Commands:", commands)
    print("Text:", text)
    print("-" * 50)
    return commands, filename

def get_video(commands, text, record_path):

    env = MinecraftSim(
        obs_size=(128, 128), 
        callbacks=[
            CommandsCallback(commands),
            JudgeResetCallback(600),
            RecordCallback(record_path=record_path, fps=30, frame_type="pov"),
        ]
    )

    model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to("cuda")
    model.eval()
    condition = model.prepare_condition(
        {
            'cond_scale': 4.0,
            'text': text
        }
    )
    state_in=model.initial_state(condition, 1)


    n = 2
    obs, info = env.reset()
    for _ in range(n): 
        memory = None
        for i in range(600):
            
            action, state_in = model.get_steve_action(condition, obs, state_in, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)

        obs, info = env.reset()
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Minecraft tasks')
    parser.add_argument('--difficulty', type=str, help='Difficulty level (simple or hard)')
    args = parser.parse_args()

    difficulty = args.difficulty
    if difficulty not in ['simple', 'hard']:
        print("Invalid difficulty level. Please choose 'simple' or 'hard'.")
        exit()

    directory = f'./task_configs/{difficulty}'

    for filename in os.listdir(directory):
        if filename.endswith('.yaml'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                yaml_content = file.read()
            
            commands, text = extract_info(yaml_content, filename)
            file_path = f"/nfs-shared-2/steve_{difficulty}/{text}"

            if os.path.exists(file_path):
                print(f"File {file_path} exists")
            else:
                text = filename[:-5]
                print("input text", text)
                get_video(commands, text, file_path)