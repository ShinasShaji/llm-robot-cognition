# LLM Robot Cognition

**Accompanying code for our paper "From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?" [[Paper website]](https://shinasshaji.github.io/icra-llm-robot-cognition/)**

This project implements a cognitive robotics system where a Large Language Model (LLM) serves as the core decision-making component for a simulated mobile manipulator operating in a household environment. The agent is embodied in a PyBullet-based simulation as a mobile manipulator with an omnidirectional base and a 7-DoF robotic arm, capable of interacting with its environment through perception, reasoning, navigation, grasping, and placement actions.

The system demonstrates how LLMs can function as cognitive controllers for robots, enabling them to perform complex household tasks that require physical interaction through navigation and object manipulation. The agent's architecture includes:
- Working memory in the form of LLM context
- Episodic memory using ChromaDB for storing past experiences
- Tool-calling interface for high-level actions

## Prerequisites

To be able to run this project, execute the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

For the LLM to function properly, you'll need API keys for one of the supported platforms:
- OpenAI API key
- Anthropic API key
- Local LLM API (e.g., vLLM)

Set these in your environment variables or create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key
# OR for local models
LOCAL_API_KEY=your_local_api_key
LOCAL_API_HOST=http://localhost:8000/v1
MODEL_NAME=name-of-local-model      # See model_loader.py to add additional models if required
```

## How to Run the Code

To run the simulation, execute the main script:

```bash
python simulation.py
```

When you run the simulation:
1. A PyBullet GUI window will open showing the environment
2. The robot will be initialized in the environment
3. An interactive command-line interface will appear where you can enter tasks for the robot

Example tasks you can give to the robot:
- "Move to the kitchen shelf"
- "Grab the mug from the table"
- "Place the box on the bottom shelf"

The robot will use its LLM-based cognitive architecture to interpret these commands and execute appropriate actions through its tool interface.

## Evaluation Setup

The evaluation pipeline runs multiple trials across configured models, tasks, and memory settings, then records results to a CSV file for analysis.

1. Configure the evaluation parameters in `config.yaml`:
   - `time_step` (simulation timestep)
   - `real_time` (run simulation in real time)
   - `headless` (disable GUI for faster batch runs)
   - `num_trials` (repetitions per task/model)
   - `num_memory_trials` (repetitions with/without memory, where memory is accumulated over the trials)
   - `models` (one or more model identifiers)
   - `prompts` (task prompts, in order of each task)
   - `conditions` (per-task success criteria, in order of each task)

2. Run the evaluation script:
   ```bash
   python evaluation.py
   ```

3. Review the output:
   - Results are appended to `results.csv`
   - Use `plot_results.py` to generate plots into the `plots/` directory

### Example Configuration

```yaml
time_step: 0.01
real_time: false
headless: true
num_trials: 20
num_memory_trials: 2
models:
   - "deepseek-v3.1"
prompts:
   - "Put away all the items into the shelf."
   - "Swap the mug and the cube in the shelf."
conditions:
   - task_index: 0
      value:
         location:
            Kitchen Shelf:
               place_positions:
                  - top
                  - middle
                  - bottom
      condition: "occupied_by"
      targets:
         - "mug"
         - "cube"
         - "box"
   - task_index: 1
      value:
         location:
            Kitchen Shelf:
               place_positions:
                  - top
                  - middle
                  - bottom
      condition: "swap"
      targets:
         - "mug"
         - "cube"
```

## Robot Capabilities and Tools

The robot has access to several tools that allow it to interact with its environment:

1. **Look around**: Returns information about the current environment, including locations and objects
2. **Move to**: Navigate to a specific location in the environment using path planning
3. **Grab**: Pick up an object by name if within reach
4. **Place**: Place a held object at a specified location
5. **Search memory**: Query past experiences stored in episodic memory
6. **Add to scratchpad**: Write down thoughts and plans for reasoning
7. **View scratchpad**: Review previous thoughts and plans
8. **End task**: Complete the current task with a status report

## Adding Objects to the World

To add new objects to the world, you can modify the `create_default_physical_objects()` method in `world.py`. Here's how to add a new object:

1. Create a visual shape for the object:
   ```python
   visual_shape = p.createVisualShape(p.GEOM_MESH, fileName="./models/YourObject.obj", meshScale=[scale, scale, scale])
   ```

2. Create a collision shape:
   ```python
   collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[x_size/2, y_size/2, z_size/2])
   ```

3. Create the multi-body object:
   ```python
   object_id = p.createMultiBody(
       baseMass=mass,
       baseCollisionShapeIndex=collision_shape,
       baseVisualShapeIndex=visual_shape,
       basePosition=[x, y, z],
       baseOrientation=p.getQuaternionFromEuler([roll, pitch, yaw])
   )
   ```

4. Add the object to the world's object dictionary:
   ```python
   self.add_object(object_id, "object_name")
   ```

## Adding New Locations

To add new locations to the semantic map:

1. Create a new Location object:
   ```python
   new_location = Location("Location Name", [x, y])
   ```

2. Add it to the world:
   ```python
   world.add_location(new_location)
   ```

3. Connect it to neighboring locations:
   ```python
   world.add_next_to("New Location", ["Existing Location 1", "Existing Location 2"])
   ```

## Memory System

The robot uses [ChromaDB](https://www.trychroma.com/) for episodic memory storage. Memories are stored as vector embeddings and can be retrieved using semantic search. The memory system allows the robot to learn from past experiences and adapt its strategies for similar future tasks.

### How Memories Are Written

Memories are recorded when the agent finishes a task using the **End task** tool. The system stores:
- **Content**: the task description provided to the robot
- **Metadata**: the task status and a summary of the execution trace

This means each completed task yields a compact, searchable “episode” that can be retrieved in later tasks via **Search memory**.

### Composition Over Task Iterations

Memories accumulate over the sequence of tasks within a run. If your `prompts` list contains multiple tasks, each successful completion is appended to the memory store, and later tasks can query those earlier episodes.

During evaluation, memory is managed per memory iteration:
- **Memory iteration 0** starts from a clean slate (existing memories are cleared).
- **Subsequent memory iterations** re-use the same memory database, so episodes from earlier iterations remain available and grow over time.

This setup makes it possible to compare “no-memory” vs. “with-memory” performance by controlling `num_memory_trials` in `config.yaml`.

## Citing

Please cite our work using the following bibtex:

```bibtex
@inproceedings{shaji_huppertz_mitrevski_houben2026,
    authors   = {Shaji, Shinas and Huppertz, Fabian and Mitrevski, Alex and Houben, Sebastian},
    title     = {{From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?}},
    booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
    year      = {2026}
}
```