import dotenv
dotenv.load_dotenv()

import os
import yaml
import csv
from pprint import pprint

from world import World
from robot import Robot
from simulation import SimulationEnvironment
from world_state_checker import WorldStateChecker
from logging_utils import get_logger
from model_loader import load_model

# Initialize logger
logger = get_logger("EVALUATION")

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logger.info(f"Loaded config:")
pprint(config)

time_step = config.get('time_step', 0.01)
real_time = config.get('real_time', True)
headless = config.get('headless', False)
num_trials = config.get('num_trials', 1)
num_memory_trials = config.get('num_memory_trials', 2)
models_config = config.get('models', [])
prompts_config = config.get('prompts', [])
conditions_config = config.get('conditions', [])

# If no models specified, run once with no model
if not models_config:
    models_config = [None]

# If no prompts specified, run once with a default prompt
if not prompts_config:
    prompts_config = ["Put away all the items into the shelf."]

# Initialize CSV file handling
file_exists = os.path.exists('results.csv')
fieldnames = ['Model', 'Memory', 'Task', 'Trial', 'Invokes', 'Toolcalls', 'Belief', 'Truth', 'Accuracy', 'Stopped']

for model_name in models_config:
    for iteration_index in range(num_trials):
        for memory_trial_index in range(num_memory_trials):
            # Create a simulation environment
            world = World.create_default_world()
            sim = SimulationEnvironment(world, time_step=time_step, real_time=real_time, headless=headless)
            sim.world.create_default_physical_objects()

            # Initialize world state checker
            state_checker = WorldStateChecker(world, conditions_config)

            # Load model if specified
            if model_name:
                logger.info(f"Loading model: {model_name}")
                model = load_model(model_name)

                if model is None:
                    continue
            else:
                model = None
                continue

            robot = Robot(sim, model)
            if robot.chat_agent is not None:
                robot.chat_agent.init_messages()

                # Delete all memories if this is the first memory iteration
                if memory_trial_index == 0:
                    robot.memory.delete_all_memories()
 
            sim.add_subscriber(robot)

            # Run the simulation
            sim.step(200)

            # Loop through prompts from config
            for task_index, task_prompt in enumerate(prompts_config):
                logger.info(f"Executing prompt {task_index + 1}/{len(prompts_config)}: {task_prompt}")

                # Record initial state before robot invocation
                state_checker.record_initial_state(task_index)

                # Skip empty prompts
                if not task_prompt.strip():
                    continue

                # Reset toolkit counters before each task
                if robot.toolkit:
                    robot.toolkit.num_toolcalls = 0
                    robot.toolkit.end_task_status = None
                    robot.num_invokes = 0

                # Invoke the robot's agent with the task prompt
                robot.invoke(task_prompt)

                results = state_checker.check_final_state(task_index)

                # Extract data for CSV using dictionary
                result_dict = {
                    'Model': robot.chat_agent.model_backend.model_type if robot.chat_agent else "None",
                    'Memory': memory_trial_index,
                    'Task': task_index + 1,  # 1-based indexing for tasks
                    'Trial': iteration_index + 1,  # 1-based indexing for trials
                    'Invokes': robot.num_invokes,
                    'Toolcalls': robot.toolkit.num_toolcalls if robot.toolkit else 0,
                    'Belief': 1 if (robot.toolkit.end_task_status == "success" if robot.toolkit else False) else 0,
                    'Truth': 1 if (results.get('status', 'error') == "success") else 0,
                    'Accuracy': 0,  # Will be calculated after getting belief and truth
                    'Stopped': 1 if robot.stopped else 0,
                }

                # Calculate accuracy (1 if belief matches truth, 0 otherwise)
                result_dict['Accuracy'] = 1 if result_dict['Belief'] == result_dict['Truth'] else 0

                # Save result to CSV immediately after each iteration
                with open('results.csv', 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Only write header if file doesn't exist or is empty
                    if not file_exists or os.path.getsize('results.csv') == 0:
                        writer.writeheader()
                        file_exists = True  # Update flag after writing header
                    
                    writer.writerow(result_dict)

                logger.info(f"State check results for task {task_index}:")
                pprint(results)
                logger.info(f"Collected data: Model={result_dict['Model']}, Memory={result_dict['Memory']}, "
                      f"Task={result_dict['Task']}, Trial={result_dict['Trial']}, "
                      f"Toolcalls={result_dict['Toolcalls']}, Belief={result_dict['Belief']}, "
                      f"Truth={result_dict['Truth']}, Accuracy={result_dict['Accuracy']}, "
                      f"Stopped={result_dict['Stopped']}"
                      )

                # Delay at end
                sim.step(100)

                if not results.get('status', 'error') == "success":
                    logger.info(f"Stopping at task index {task_index} as robot was unsuccessful...")
                    break

            # Disconnect from PyBullet
            sim.disconnect()
