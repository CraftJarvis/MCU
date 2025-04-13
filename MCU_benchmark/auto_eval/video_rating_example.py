import cv2  # We're using OpenCV to read video
import base64
import time
from openai import OpenAI
import os
import requests
import shutil
from PIL import Image
from io import BytesIO
import pdb
import json
import datetime


path_list = [
    {
        'mcu_path': '/nfs-shared-2/result/steve_hard',
        'out_dir': './steve_hard_res'
    },
    {
        'mcu_path': '/nfs-shared-2/result/steve_simple',
        'out_dir': './steve_simple_res'
    },
    {
        'mcu_path': '/nfs-shared-2/result/vpt_bc_hard',
        'out_dir': './vpt_bc_hard_res'
    },
    {
        'mcu_path': '/nfs-shared-2/result/vpt_bc_simple',
        'out_dir': './vpt_bc_simple_res'
    },
    {
        'mcu_path': '/nfs-shared-2/result/vpt_simple_rl',
        'out_dir': './vpt_rl_simple_res'
    },
    {
        'mcu_path': '/nfs-shared-2/result/vpt_hard_rl',
        'out_dir': './vpt_rl_hard_res'
    }
]


with open('./single_rating_prompt.txt', 'r', encoding='utf-8') as file:  
    system_content = file.read()

def fetch_gpt4(query):
    print('fetching gpt4 ...')
    client = OpenAI(api_key='',
                    base_url = '')
    completion = client.chat.completions.create(
        model="gpt-4o", #gpt-4o-mini gpt-4o
        messages=query,
        temperature=0.5
    )
    res = completion.choices[0].message.content
    return res

def assess_video(task_name, frames, video_path_a):
    try:
        with open(f'./criteria_files/{task_name}.txt', 'r', encoding='utf-8') as file:  
            grading_rule = file.read()
    except:
        print("no task file")
        return None
    query = [
        {
        "role": "system",
        "content": system_content
        },
        {
        "role": "user", "content":  
        f'The task name is ' + task_name + ' '
        + f'You should follow the following grading criteria to score the performance of agents in videos' + grading_rule +'\n'
        + f'Here are the image frames of the video A '
        }]

    query.append({"role": "user", "content": [{
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{frame}"
          },
        } for frame in frames
        ]})
    ans = fetch_gpt4(query)
    print(ans)
    save_data_json(ans, video_path_a, task_name)
    answer = {"role": "assistant", "content": f'{ans}'}
    return ans

def save_data_json(ans, video_path_a, task_name):
    result_dict= {}
    keys_to_extract = [  
        "Task Progress",  
        "Action Control",  
        "Error Recognition and Correction",  
        "Creative Attempts",  
        "Task Completion Efficiency",  
        "Material Selection and Usage"  
    ]  

    for line in ans.strip().split('\n'):  
        for key in keys_to_extract:  
            if line.startswith(f'- {key}: '):  
                value = (line.split(': ', 1)[1].strip()) 
                
                if value: 
                    result_dict[key] = value  
                    break  
    result_dict['video_path'] = video_path_a
    result_dict['task_name'] = task_name
    
    out_file = os.path.join(out_dir, json_name)
    with open(out_file, 'w') as f:
        json.dump(result_dict, f, indent = 4)

    return result_dict  


def process_video(task_name, video_path):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    base64Frames1 = base64Frames[0::30]
    print(len(base64Frames1), "frames read.")
    if (len(base64Frames1)>60):
        base64Frames1 = base64Frames[0::70]


    return base64Frames1

def find_mp4_files(directory):
    mp4_files = []
    task_list = []
    video_name = []
    for root, dirs, files in os.walk(directory):
        files = files[:5]
        for file in files:
        
            if file.endswith('.mp4'):

                parent_dir = os.path.basename(root)
                full_path = os.path.join(root, file)
                video_name.append(file)
                mp4_files.append(full_path)
                task_list.append(parent_dir)
    return mp4_files, task_list, video_name


def main():
  
    for path_info in path_list:
        global mcu_path
        global out_dir
        mcu_path = path_info['mcu_path']
        out_dir = path_info['out_dir']
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        files_and_dirs = os.listdir(out_dir)
        mp4_files, task_list, video_name = find_mp4_files(mcu_path)


        for i in range(len(mp4_files)):
            video_path_a = mp4_files[i]

            task_name = task_list[i]
            global json_name
            json_name = task_name+video_name[i][:-4]+'.json'
            if json_name in files_and_dirs:
                continue
            if "from_scratch" in task_name:
                continue
            print(task_name, video_path_a)
            video_a = process_video(task_name, video_path_a)
            task_name = task_name.replace(' ', '_')
            # pdb.set_trace()
            assess_video(task_name, video_a, video_path_a)


video_path_a = None
json_name = None

main()
