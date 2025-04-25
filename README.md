
# ⛏️ MCU-turbo: A Standard Benchmark for Evaluating Minecraft Agents

**MCU-turbo** is a standard benchmark based on the ![MCU framework](https://github.com/grandsmile/MCU/tree/master/figs/mcu.png), which originally features over 3000+ atomic tasks. This benchmark is designed to be a standard test, selecting **80 handcrafted atomic tasks** across 10 categories. Each task is evaluated under **two difficulty levels**—**Simple** and **Hard**—to rigorously test agent generalization, tool use, planning, and robustness under environmental variations.

> 🔍 Simple mode: Tasks begin with sufficient necessary resources pre-supplied and a clear environment.  
> 🌪️ Hard mode: Agents face **limited resources** and **disruptive factors** such as poor visibility (e.g. bad weather, night-time), extra distractors (e.g., swarms of mobs, scattered items). 

---

## 🧠 Benchmark Highlights

- 🧩 **Diverse Domains**: 80 atomic tasks and 20 compositional tasks across **combat**, **crafting**, **mining**, **creative building**, and more.
- 🔄 **Dual Difficulty**: Each task runs in both **simple** and **hard** versions to evaluate **intra-task generalization**.
- 📦 **Agent-Agnostic**: Compatible with [MineStudio](https://github.com/CraftJarvis/MineStudio) agents or any API-based Minecraft wrapper.
- 🎯 **VLM-based Evaluation**: A vision-language model analyzes video trajectories using multi-dimensional criteria.

---

## 🧪 Task Overview

Below is a curated subset of tasks from the full set of 80, organized by category. Tasks marked with 🌕 and 🌑 indicate presence in both simple and hard modes.

> 📂 *All tasks include executable task configs in* `/MCU/MCU_benchmark/task_configs`.
> 📊 The analysis of our baseline results can be found in `/MCU/docs/baseline.md`.


### ⚔️ Combat

| Task | Description |
|------|-------------|
| `combat_enderman` 🌕🌑 | combat and kill Endermen |
| `combat_skeletons` 🌕🌑 | combat and kill skeletons |
| `combat_spiders` 🌕🌑 | combat and kill multiple spiders |
| `combat_zombies` 🌕🌑 | combat and kill zombies |
| `combat_witch` 🌕🌑 | combat and kill a witch |
| `combat_wolfs` 🌕🌑 | combat and defeat wolves |
| `hunt_pigs` 🌕🌑 | hunt pigs with a sword |
| `hunt_horse` 🌕🌑 | hunt a horse with a bow or sword |
| `shoot_phantom` 🌕🌑 | shoot phantoms with a bow and arrows |

---

### 🛠️ Crafting

| Task | Description |
|------|-------------|
| `craft_enchanting_table` 🌕🌑 | craft an enchantment table |
| `craft_ladder` 🌕🌑 | craft a ladder |
| `craft_smelting` 🌕🌑 | craft a furnace for smelting |
| `craft_stonecut` 🌕🌑 | craft a stone cutter |
| `craft_the_crafting_table` 🌕🌑 | craft a crafting table |
| `craft_to_cake` 🌕🌑 | craft a cake |
| `craft_to_clock` 🌕🌑 | craft a clock |
| `craft_diorite` 🌕🌑 | craft a diorite |
| `craft_bee_nest` 🌕🌑 | craft a bee nest |
| `craft_oak_planks` 🌕🌑 | craft oak planks |

---

### 🧰 Tool Use

| Task | Description |
|------|-------------|
| `carve_pumpkins` 🌕🌑 | carve pumpkins using shears |
| `sleep_in_bed` 🌕🌑 | sleep in a bed |
| `smelt_beef` 🌕🌑 | smelt raw beef into steak |
| `drink_harming_potion` 🌕🌑 | drink a harming potion |
| `make_fire_with_flint_and_steel` 🌕🌑 | make fire using flint and steel |
| `use_bow` 🌕🌑 | use bow as your weapon |
| `use_lead` 🌕🌑 | use lead to get animals |
| `use_trident` 🌕🌑 | use trident to hunt animal |
| `use_shield` 🌕🌑 | defend yourself using a shield |
| `plant_wheats` 🌕🌑 | plant wheat seeds on farmland |

---

### ⛏️ Mining & Collecting

| Task | Description |
|------|-------------|
| `mine_diamond_ore` 🌕🌑 | mine diamond ore with an iron pickaxe |
| `mine_horizontally` 🌕🌑 | mine horizontally through a line of blocks |
| `mine_iron_ore` 🌕🌑 | mine iron ore with a stone pickaxe |
| `mine_obsidian` 🌕🌑 | mine obsidian with a diamond pickaxe |
| `mine_dirt` 🌕🌑 | mine dirt using a wooden shovel |
| `mine_grass` 🌕🌑 | mine grass blocks using a shovel |
| `mine_wood` 🌕🌑 | mine oak logs with a wooden axe |
| `collect_wool` 🌕🌑 | collect wool from sheep using shears |

---

### 🧠 Creative

| Task | Description |
|------|-------------|
| `prepare_a_birthday_present_for_your_neighbor` 🌕🌑 | prepare a birthday present for your neighbor |

---

### 🏗️ Building

| Task | Description |
|------|-------------|
| `build_nether_portal` 🌕🌑 | build a nether portal |
| `build_gate` 🌕🌑 | build a gate using a crafting table |
| `build_pillar` 🌕🌑 | build a pillar with cobblestone blocks |
| `build_snow_golem` 🌕🌑 | build a snow golem |
| `build_a_house` 🌕🌑 | build a simple house |
| `build_a_wall` 🌕🌑 | build a wall using stone bricks |
| `build_a_ladder` 🌕🌑 | craft a ladder using sticks |
| `build_a_tower` 🌕🌑 | build a tower using available materials |
| `build_a_waterfall` 🌕🌑 | build a waterfall using water buckets and stone |
| `build_a_library` 🌕🌑 | build a library using bookshelves and wood planks |
| `dig_three_down_and_fill_one_up` 🌕🌑 | dig three blocks down and fill one block up |
| `build_a_garden` 🌕🌑 | build a garden using various blocks |
| `build_a_maze` 🌕🌑 | construct a simple maze using stone blocks |

---

### 🖼️ Decoration

| Task | Description |
|------|-------------|
| `decorate_the_ground` 🌕🌑 | decorate the ground using various blocks and items |
| `clean_the_weeds` 🌕🌑 | clean the weeds using a hoe |
| `lay_carpet` 🌕🌑 | lay a carpet on the ground |
| `decorate_the_wall` 🌕🌑 | decorate a wall using various decorations |
| `light_up_the_surroundings` 🌕🌑 | light up the surroundings |
| `place_a_item_frame` 🌕🌑 | place an item frame on a block |

---

### 🌀 Motion

| Task | Description |
|------|-------------|
| `look_at_the_sky` 🌕🌑 | look at the sky |
| `drop_an_item` 🌕🌑 | drop an item from your inventory |
| `stacking_acacia_fence` 🌕🌑 | stack acacia fences |
| `throw_a_snowball` 🌕🌑 | throw a snowball |

---

### 🧲 Finding

| Task | Description |
|------|-------------|
| `find_bedrock` 🌕🌑 | find bedrock |
| `find_lava` 🌕🌑 | find a lava pool |
| `find_sand` 🌕🌑 | find sand |
| `find_blue_bed` 🌕🌑 | find a blue bed |
| `find_item_frame` 🌕🌑 | find an oak door |
| `find_diamond` 🌕🌑 | find and mine diamond ore |
| `find_melon` 🌕🌑 | find a melon |
| `find_forest` 🌕🌑 | find a forest using a map |
| `find_village` 🌕🌑 | find a village using a map |

---

### 🧭 Exploration

| Task | Description |
|------|-------------|
| `explore_boat` 🌕🌑 | explore with a boat on water |
| `explore_chest` 🌕🌑 | explore the contents of a chest |
| `explore_climb` 🌕🌑 | explore and climb a mountainous terrain |
| `explore_run` 🌕🌑 | explore and run |
| `explore_map` 🌕🌑 | explore with a map |

---

### 🪤 Trapping

| Task | Description |
|------|-------------|
| `trap_a_spider` 🌕🌑 | trap a spider with a boat |
| `trap_a_witch` 🌕🌑 | trap a witch with a boat |
| `hook_a_chicken` 🌕🌑 | hook a chicken using a fishing rod |
| `hook_a_cow` 🌕🌑 | hook a cow using a fishing rod |

---

## 🛠️ Environment Setup

1. Clone this repo:

```bash
git clone https://github.com/YOUR_USERNAME/MCU.git
cd MCU
```

2. Install dependencies:


```bash
conda create -n mcu python=3.10 -y
conda activate mcu
conda install --channel=conda-forge openjdk=8 -y
pip install MineStudio
```

---

## 🧪 Evaluation

Run tasks:

```bash
cd MCU_benchmark
python run_task.py \
  --difficulty simple 
```

Evaluation video are automatically saved in `output/`.

VLM evaluation:

```bash
cd auto_eval
python batch_video_rating.py \
  --videos_path='./output/' \
  --criteria_files_path='./auto_eval/criteria_files/' 
```



---

## 📤 Contribute

You can contribute new tasks or difficulty configurations. Submit PRs or open issues to discuss!