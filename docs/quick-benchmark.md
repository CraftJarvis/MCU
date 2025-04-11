
# Quick benchmark

This document outlines the structure and workflow of the benchmark module, a comprehensive framework designed for evaluating agent performance in Minecraft environments.

## Code structure

Below is the structure of the benchmark module, which organizes task definitions and testing scripts for evaluation:

```plaintext
benchmark/
    ├── task_configs/ 
    │   ├── simple/ 
    │   │   └── Task definitions for simple tasks.
    │   ├── hard/
    │       └── Task definitions for complex tasks.
    ├── test_pipeline.py
    │   └── Example script for parallelized and batched task execution.
    ├── test.py
    │   └── Example script for running batch tests.
    ├── utility/
    │   └── Functionality for input reading and callback features.
```


## Workflow Overview

### Task Configuration

Tasks are defined in YAML files located in the `task_configs/` directory. under the appropriate difficulty subdirectory (e.g., `simple/` or `hard/`).
 Example YAML:

```yaml
custom_init_commands: 
- /give @s minecraft:water_bucket 3
- /give @s minecraft:stone 64
- /give @s minecraft:dirt 64
- /give @s minecraft:shovel{Enchantments:[{id:"minecraft:efficiency",lvl:1}]} 1
text: Build a waterfall in your Minecraft world.
```

Key Elements of the YAML File:

1. **`custom_init_commands`**:
   - Specifies commands to initialize the Minecraft environment for the task.
   - Examples:
     - `/give @s minecraft:water_bucket 3`: Gives the agent three water buckets.
     - `/give @s minecraft:stone 64`: Provides a stack of stone blocks.
   - These commands ensure the agent has the necessary tools and resources to perform the task.

2. **`text`**:
   - Provides a natural language description of the task.
   - Example: `"Build a waterfall in your Minecraft world."`


### Running Tests

1. **Individual or Small-Scale Tests**:
   - Use `test.py` for running specific tasks or testing new configurations.
     ```console
     $ python test.py
     ```

2. **Batch Testing with Parallelization**:
   - Use `test_pipeline.py` for executing tasks in parallel.
     ```console
     $ python test_pipeline.py
     ```
---

#### An Example: `test.py`

This script demonstrates how to evaluate tasks using YAML-based configurations. Below is an outline of its workflow:

1. **Task Setup**:
   - Load configuration files from `task_configs/simple`.
   - Parse YAML files into callbacks using `convert_yaml_to_callbacks`.

2. **Environment Initialization**:
   - Use `MinecraftSim` to create a simulation environment.
   - Add callbacks:
     - `RecordCallback`: Saves video frames for evaluation.
     - `CommandsCallback`: Initializes the environment.
     - `TaskCallback`: Implements task-specific behavior.

3. **Task Execution**:
   - Reset the environment and run the task for multiple steps.
   - Save observations, actions, and outputs for analysis.

4. **Result Storage**:
   - Videos and logs are saved in the `output/` directory.


```python
commands_callback, task_callback = convert_yaml_to_callbacks("./task_configs/simple/build_waterfall.yaml")
env = MinecraftSim(
   obs_size=(128, 128), 
   callbacks=[
         RecordCallback(record_path=f"./output/", fps=30, frame_type="pov"),
         CommandsCallback(commands_callback),
         TaskCallback(task_callback),
   ]
)
policy = load_vpt_policy(
   model_path="pretrained/foundation-model-2x.model",
   weights_path="pretrained/foundation-model-2x.weights"
).to("cuda")

obs, info = env.reset()
for i in range(12000):
   action, memory = policy.get_action(obs, memory, input_shape='*')
   obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

