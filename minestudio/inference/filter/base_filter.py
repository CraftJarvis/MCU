'''
Date: 2024-11-25 07:36:18

LastEditTime: 2024-11-25 12:33:00
FilePath: /MineStudio/minestudio/inference/filter/base_filter.py
'''
from abc import abstractmethod
from typing import List, Dict, Generator

class EpisodeFilter:

    def __init__(self):
        pass

    def filter(self, episode_generator: Generator) -> Generator:
        return episode_generator