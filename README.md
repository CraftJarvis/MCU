Based on your task list and the MCU/KUMO style, here's a stylized and clean draft for your **README**. It introduces your custom benchmark, aligns with the structure of KUMO, and adds sections for categories and difficulty modes (simple and hard):

---

# ⛏️ MCU: A Benchmark for Evaluating Minecraft Agent

**MCU** is an extended benchmark based on the [MCU framework](https://arxiv.org/abs/2504.02810), featuring **80 handcrafted atomic tasks** across 10 categories. Each task is evaluated under **two difficulty levels**—**Simple** and **Hard**—to rigorously test agent generalization, tool use, planning, and robustness under environmental variations.

> 🔍 Simple mode: Tasks begin with all necessary items pre-supplied and a clear environment.  
> 🌪️ Hard mode: Agents face **disruptive factors** such as night-time settings, poor visibility, extra distractors (e.g. mobs), or misleading initial setups.

---

## 🧠 Benchmark Highlights

- 🧩 **Diverse Domains**: 80 atomic tasks across **combat**, **crafting**, **mining**, **creative building**, and more.
- 🔄 **Dual Difficulty**: Each task runs in both **simple** and **hard** versions to evaluate **intra-task generalization**.
- 🎯 **Grounded Evaluation**: Tasks are executable and verified via a standardized environment config.
- 📦 **Agent-Agnostic**: Compatible with [MineStudio](https://github.com/open-mine/mine-studio) agents or any API-based Minecraft wrapper.

---

## 🧪 Task Overview

Below is a curated subset of tasks from the full set of 80, organized by category. Tasks marked with 🌕 and 🌑 indicate presence in both simple and hard modes.

### ⚔️ Combat

| Task | Description |
|------|-------------|
| `combat_enderman` 🌕🌑 | Locate and defeat an Enderman |
| `combat_skeletons` 🌕🌑 | Survive a skeleton ambush at night |
| `combat_phantom` 🌕🌑 | Fight off flying phantoms from above |
| `hunt_pigs` 🌕🌑 | Track and kill pigs for food |
| `combat_witch` 🌕🌑 | Eliminate a hostile witch using tactics |

---

### 🛠️ Crafting

| Task | Description |
|------|-------------|
| `craft_to_cake` 🌕🌑 | Craft a cake from scratch using proper ingredients |
| `craft_enchantment` 🌕🌑 | Build an enchanting table |
| `craft_the_crafting_table` 🌕🌑 | Basic wood-to-crafting-table logic |
| `craft_clock` 🌕🌑 | Craft a clock using mined gold and redstone |
| `enchant_diamond_sword` 🌕🌑 | Enchant a sword using XP and a book table |

---

### ⛏️ Mining

| Task | Description |
|------|-------------|
| `mine_diamond_ore` 🌕🌑 | Find and mine diamond ores |
| `mine_iron_ore` 🌕🌑 | Locate and extract iron |
| `collect_wood` 🌕🌑 | Harvest logs from nearby trees |
| `mine_obsidian` 🌕🌑 | Mine obsidian using correct tools |
| `collect_dirt` 🌕🌑 | Gather dirt blocks efficiently |

---

### 🧰 Tool Use

| Task | Description |
|------|-------------|
| `use_bow` 🌕🌑 | Hit a target using a bow |
| `smelt_beef` 🌕🌑 | Cook raw beef using a furnace |
| `make_fire_with_flint_and_steel` 🌕🌑 | Ignite surroundings |
| `use_trident` 🌕🌑 | Throw a trident at a mob |
| `plant_wheats` 🌕🌑 | Till land and plant seeds |

---

### 🏗️ Building

| Task | Description |
|------|-------------|
| `build_nether_portal` 🌕🌑 | Construct a valid portal frame |
| `build_snow_golem` 🌕🌑 | Assemble a snow golem from materials |
| `build_house` 🌕🌑 | Construct a basic wooden house |
| `build_wall` 🌕🌑 | Build a defensive wall |
| `build_tower` 🌕🌑 | Create a vertical watchtower |

---

### 🎨 Decoration

| Task | Description |
|------|-------------|
| `decorate_the_ground` 🌕🌑 | Place carpets and flowers aesthetically |
| `decorate_the_wall` 🌕🌑 | Use item frames and torches on walls |
| `clean_the_weeds` 🌕🌑 | Remove unwanted vegetation |
| `place_item_frame` 🌕🌑 | Frame an item on the wall |

---

### 🧭 Exploration

| Task | Description |
|------|-------------|
| `explore_boat` 🌕🌑 | Navigate across a river in a boat |
| `explore_chest` 🌕🌑 | Discover and open a hidden chest |
| `explore_climb` 🌕🌑 | Reach the top of a mountain |
| `explore_run` 🌕🌑 | Sprint to a distant location |
| `explore_map` 🌕🌑 | Use a map to locate a point of interest |

---

### 🔎 Finding

| Task | Description |
|------|-------------|
| `find_diamond` 🌕🌑 | Locate a diamond ore block |
| `find_village` 🌕🌑 | Discover a nearby village |
| `find_blue_bed` 🌕🌑 | Locate a specific colored bed |
| `find_oak_door` 🌕🌑 | Find a house with an oak door |

---

### 🪤 Trapping

| Task | Description |
|------|-------------|
| `trap_a_spider` 🌕🌑 | Use fences to contain a spider |
| `trap_a_witch` 🌕🌑 | Safely capture a witch |
| `hook_a_chicken` 🌕🌑 | Use a lead or fishing rod on a chicken |

---

### 🧠 Creative / Puzzle

| Task | Description |
|------|-------------|
| `prepare_a_birthday_present` 🌕🌑 | Make a surprise setup for a villager |
| `dig_three_down_fill_one_up` 🌕🌑 | Execute an abstract digging pattern |
| `build_a_maze` 🌕🌑 | Construct a small navigable maze |

---

## 🛠️ Environment Setup

1. Clone this repo:

```bash
git clone https://github.com/YOUR_USERNAME/MCU.git
cd MCU
```

2. Install dependencies:

```bash
conda create -n mcux python=3.12
conda activate mcux
pip install -r requirements.txt
```

---

## 🧪 Evaluation

Run a task:

```bash
python run_task.py \
  --task_name craft_to_cake \
  --difficulty simple \
  --agent_path ./agents/groot
```

Evaluation results are automatically saved in `results/`.

---

## 📤 Contribute

You can contribute new tasks or difficulty configurations. Submit PRs or open issues to discuss!

---

Would you like me to generate the JSON configs or task template files for these tasks too?