'''
Date: 2024-12-06 16:42:49

LastEditTime: 2024-12-11 17:44:10
FilePath: /MineStudio/minestudio/benchmark/utility/read_conf.py
'''
import os
import yaml

def convert_yaml_to_callbacks(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    commands = data.get('custom_init_commands', [])

    text = data.get('text', '')
    task_name = os.path.splitext(os.path.basename(yaml_file))[0]
    task_dict = {}
    task_dict['name'] = task_name
    task_dict['text'] = text

    return commands, task_dict
