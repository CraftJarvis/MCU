'''
Date: 2024-11-25 12:39:01

LastEditTime: 2024-11-25 13:19:22
FilePath: /MineStudio/minestudio/inference/filter/info_base_filter.py
'''
import pickle
from minestudio.inference.filter.base_filter import EpisodeFilter

class InfoBaseFilter(EpisodeFilter):
    
    def __init__(self, key: str, val: str, num: int, label: str = "status"):
        self.key = key
        self.val = val
        self.num = num
        self.label = label
    
    def filter(self, episode_generator):
        for episode in episode_generator:
            info = pickle.loads(open(episode["info_path"], "rb").read())
            if info[-1][self.key].get(self.val, 0) >= self.num:
                episode[self.label] = "yes"
            yield episode