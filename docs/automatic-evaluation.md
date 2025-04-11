# Automatic Evaluation Pipeline

The pipeline automates evaluation tasks in the MineStudio framework, enabling the generation of criteria and video evaluation for agent performance analysis.

---


## Code Structure

The following is the structure of the evaluation module, with each file and folder serving specific purposes:

```plaintext

auto_eval/
    ├── criteria_files/
    │   └── Contains criteria files for evaluating videos for each task.
    ├── eval_video/
    │   └── Stores example videos and provides a structure for saving task-specific evaluation videos.
    ├── batch_video_rating.py
    │   └── Batch evaluation of videos for task performance.
    ├── individual_video_rating.py
    │   └── Evaluate videos individually for detailed analysis.
    ├── video_comparison.py
        └── Compare videos to measure performance differences.
```


## Evaluating Videos with Vision-Language Models

### An Example: Comparing Videos Using `video_comparison.py`

Below is a simplified guide to using `video_comparison.py` for video comparison:

1. **Prepare Videos:**
   Ensure your video files (e.g., `video_a.mp4` and `video_b.mp4`) are placed in the `eval_video/` directory.

2. **Define Criteria:**
   Define task-specific criteria files in `criteria_files/` (e.g., `build_gate.txt`).

3. **Run the Script:**
   Use the following command to compare two videos:
   ```bash
   python video_comparison.py \
     --video_path_a='./eval_video/build_gate/build_gate_5.mp4' \
     --video_path_b='./eval_video/build_gate/build_gate_7.mp4' \
     --criteria_path='./auto_eval/criteria_files/build_gate.txt'
   ```

4. **Analyze Results:**
   After running the script, the evaluation results will be saved as a JSON file in the `vlm_rating_res/` directory.


The following is an **example output**, showcasing how two videos are compared across several evaluation criteria. Each criterion is explained with observations, and an overall assessment is provided.


```json
[
    {
        "Task Progress": "B is better",
        "Action Control": "B is better",
        "Error Recognition and Correction": "B is better",
        "Creative Attempts": "tie",
        "Task Completion Efficiency": "B is better",
        "Material Selection and Usage": "tie",
        "video_1_path": "./eval_video/build_gate_5.mp4",
        "video_2_path": "./eval_video/build_gate_7.mp4"
    },
    "Task Progress:\n- Video B constructs two pillars and an arch; A does not complete the arch.\nresult: B is better\n\nAction Control:\n- Video A shows more wandering and redundant actions.\nresult: B is better\n\nError Recognition and Correction:\n- Video B corrects structure misalignments.\nresult: B is better\n\nCreative Attempts:\n- Neither video shows creative elements like decorations.\nresult: tie\n\nTask Completion Efficiency:\n- Video B completes the task faster and more efficiently.\nresult: B is better\n\nMaterial Selection and Usage:\n- Both use oak planks appropriately.\nresult: tie\n"
]
```

1. **Assessment Dimensions**:
   - **Task Progress**: Measures how much of the task is completed.
   - **Action Control**: Assesses movement precision and avoidance of redundant actions.
   - **Error Recognition and Correction**: Evaluates the agent’s ability to detect and fix mistakes.
   - **Creative Attempts**: Considers innovative or decorative efforts beyond task requirements.
   - **Task Completion Efficiency**: Tracks speed and resourcefulness in completing the task.
   - **Material Selection and Usage**: Ensures appropriate materials are used.

2. **Structured Results**:
   - The first section provides a concise summary of the evaluation for each criterion.
   - Example:
     - `"Task Progress": "B is better"`
     - `"Creative Attempts": "tie"`

3. **Detailed Observations**:
   - The second section explains the reasoning behind each result.
   - Example:
     - **Task Progress**: "Video B constructs two pillars and an arch; A does not complete the arch."
     - **Creative Attempts**: "Neither video shows creative elements like decorations."




### Organizing Files for Batch Evaluation

Evaluate all videos in a directory using their respective criteria.

```bash
python batch_video_rating.py \
  --videos_path='./eval_video/' \
  --criteria_files_path='./auto_eval/criteria_files/'
```

Organize your task-specific videos under the `videos_path` directory:

```
videos_path     
├── build_waterfall     # task_name_1     
│     ├── episode_1.mp4
│     ├── episode_2.mp4
├── build_house         # task_name_2
│     ├── episode_1.mp4
│     ├── episode_2.mp4
├── task_name_3
│     ├── episode_1.mp4
│     ├── episode_2.mp4
```

Store criteria files under the `criteria_files_path` directory, matching the task names:

```
criteria_files_path     
├── build_waterfall.txt # task_name_1     
├── build_house.txt     # task_name_2
├── task_name_3.txt
```


### Example Commands

To evaluate task performance using pre-recorded videos and criteria, you can use the following commands depending on your needs:

- **Compare Two Videos:**

   Compare two videos of the same task to analyze differences in agent performance.  

   ```bash
   python video_comparison.py \
     --video_path_a='./eval_video/build_gate/build_gate_5.mp4' \
     --video_path_b='./eval_video/build_gate/build_gate_7.mp4' \
     --criteria_path='./auto_eval/criteria_files/build_gate.txt'
   ```

- **Individual Video Evaluation:**

   Evaluate a single video against predefined criteria.

   ```bash
   python individual_video_rating.py \
     --video_path='./eval_video/build_gate/build_gate_5.mp4' \
     --criteria_path='./auto_eval/criteria_files/build_gate.txt'
   ```

- **Batch Video Evaluation:**

   Evaluate all videos in a directory using their respective criteria.

   ```bash
   python batch_video_rating.py \
     --videos_path='./eval_video/' \
     --criteria_files_path='./auto_eval/criteria_files/'
   ```


This tutorial covers the essentials of setting up and running the automatic evaluation pipeline. For more advanced usage, explore the provided code files for customization options.