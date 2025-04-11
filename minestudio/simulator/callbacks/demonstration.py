import random
import numpy as np
import os

from minestudio.simulator.callbacks.callback import MinecraftCallback

def download_reference_videos():
    import huggingface_hub
    local_dir = os.path.join(os.path.dirname(__file__), "reference_videos")
    print(f"Downloading reference videos to {local_dir}")
    huggingface_hub.snapshot_download(repo_id='CraftJarvis/MinecraftReferenceVideos', repo_type='dataset', local_dir=local_dir)

class DemonstrationCallback(MinecraftCallback):
    """
        This callback is used to provide demonstration data, mainly for GROOT.
    """
    def __init__(self, task):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "reference_videos")):
            response = input("Detecting missing reference videos, do you want to download them from huggingface (Y/N)?\n")
            while True:
                if response == 'Y' or response == 'y':
                    download_reference_videos()
                    break
                elif response == 'N' or response == 'n':
                    break
                else:
                    response = input("Please input Y or N:\n")

        self.task = task

        # load the reference video
        ref_video_name = task

        assert os.path.exists(os.path.join(os.path.dirname(__file__), "reference_videos", ref_video_name)), f"Reference video {ref_video_name} does not exist."

        ref_video_path = os.path.join(os.path.dirname(__file__), "reference_videos", ref_video_name, "human")

        # randomly select a video end with .mp4
        ref_video_list = [f for f in os.listdir(ref_video_path) if f.endswith('.mp4')]

        ref_video_path = os.path.join(ref_video_path, random.choice(ref_video_list))

        self.ref_video_path = ref_video_path

    def after_reset(self, sim, obs, info):
        obs['ref_video_path'] = self.ref_video_path
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs['condition'] = self.ref_video_path
        return obs, reward, terminated, truncated, info
