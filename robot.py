import backoff
import pybullet as p
import numpy as np

from camel.agents import ChatAgent
from typing import Dict, TYPE_CHECKING

from toolkit import RobotToolkit
from memory import Memory
from logging_utils import get_logger

# Initialize logger
logger = get_logger("ROBOT")

if TYPE_CHECKING:
    from simulation import SimulationEnvironment


class Robot:
    def __init__(self, simulation_env: 'SimulationEnvironment', model=None):
        self.env = simulation_env
        self.env.add_subscriber(self)

        # Initialize robot components
        # Create mobile base (cube)
        base_size = [0.7, 0.7, 0.3]
        base_start_pos = [0.0, 0.0, base_size[2] / 2]
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in base_size])
        self.base_id = p.createMultiBody(baseMass=1000.0, baseCollisionShapeIndex=base_collision,
                                    baseVisualShapeIndex=base_visual, basePosition=base_start_pos)

        # Load robot arm on top of base
        # Adjust start position to be on top of the cube base
        robot_start_pos = [base_start_pos[0], base_start_pos[1], base_size[2]]
        self.robot_id = p.loadURDF("models/iiwa.urdf", robot_start_pos, useFixedBase=False)

        p.resetBasePositionAndOrientation(self.robot_id, robot_start_pos,[0.0,0.0,0.0,1.0])
        # Attach arm to base using fixed constraint
        p.createConstraint(
            parentBodyUniqueId=self.base_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.robot_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=robot_start_pos,
            childFramePosition=base_start_pos
        )

        self.constraint_id = None
        self.held_object_id = None

        # Movement state
        self.action_target = None
        self.activity = set()

        # Track invokes
        self.num_invokes = 0
        self.max_num_invokes = 10
        self.stopped = False

        # Path planning and following
        self.path = []
        self.waypoint_index = 0
        self._reset_path()

        # Agent
        system_message = """You are a robot assistant. Your goal is to perform tasks given to you.

You have the following tools available to you to assist with tasks:
- look_around: Look around and return information about the environment. Use this tool when you need to get information about your current surroundings, including locations and objects.
- move_to: Move the robot base to a target location by name (str). If you need to place or pick something, consider moving in front of the object in question before doing so, if such a location exists. A planner is available to navigate through intermediate locations, so providing the name of the intended final target is sufficient.
- grab: Pick up an object by name (str). You must move to the location containing the object (or in front of the object) first before grabbing it or grabbing from it. Successfully grabbing an object makes the object the currently held object.
- place: Place the held object at the given location by name (str). You must move to the location (or in front of the location) first before placing the object there. Note that you must have a currently held item that you can place. Successfully placing an object will remove it from being the currently held object.
- end_task: End the current task with a status report. Use this tool when you have completed the task or determined that it cannot be completed. Provide a status (success, failure, or unknown), a description of the original task including its status, and a detailed summary of the execution trace. Be sure to provide all information about the task execution, not missing details on any steps of the task execution. Once the task is ended, you will receive a new task.
- search_memory: Search your memory for previously completed tasks that might be relevant to the current task. Use this tool when you need information from past experiences to help with the current task. Always use this tool to check for previous experiences when starting a task.
- add_to_scratchpad: Add an entry to your scratchpad for reasoning and reflection. Use this tool when you want to think through a problem and plan actions or record your thoughts before taking action. The scratchpad is private to you.
- view_scratchpad: View the current contents of your scratchpad joined as paragraphs. Use this tool to review your previous thoughts and reasoning.

If you come across issues or ambiguities, think in detail about what may have caused them, and take alternative approaches or measures to complete the task. Be agentic, and have a problem-solving approach to performing the task at hand. You should perform tasks with minimal additional supervision.

Guidelines for optimal execution:
- When given an unambiguous request, reason about reasonable ways to perform the task, and do not take things too literally.
- Every time you need to perform a task, query your memory to see if you have done the task before, as well as to see how you mitigated issues that you may encounter in the task.
- Use your scratchpad to make a plan, reason about your plan, as well as issues you may face in detail before making actions.
- Ensure that you are in the correct location near an object before attempting to grab the object.
- Ensure that you are in the correct location to place an object, before attempting to place the object.
"""
        if model:
            self.memory = Memory()
            self.toolkit = RobotToolkit(self, self.memory)
            self.chat_agent = ChatAgent(
                system_message=system_message,
                model=model,
                tools=self.toolkit.get_tools(),
            )
            self.first_turn = True
        else:
            self.toolkit = None
            self.chat_agent = None

    @property
    def position(self):
        pos, ori = p.getBasePositionAndOrientation(self.base_id)
        return pos

    @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def _step_with_backoff(self, **kwargs):
        if self.chat_agent:
            return self.chat_agent.step(**kwargs)
    
    def _step(self, message):
        """Internal method to handle agent stepping with optional streaming."""
        # Check if streaming is enabled
        # Try to access model configuration similar to reference code
        stream = self.chat_agent.model_backend.model_config_dict.get("stream", False)

        if stream:
            response = None
            previous_length = 0
            consecutive_newlines = 0
            logger.info("Generating streaming response...")
            for i, response in enumerate(self.chat_agent.step(message)):
                if msgs := getattr(response, "msgs"):
                    current_content = msgs[0].content
                    new_content = current_content[previous_length:]
                    
                    # Limit consecutive newlines to 2
                    filtered_content = ""
                    for char in new_content:
                        if char == '\n':
                            if consecutive_newlines < 2:
                                filtered_content += char
                                consecutive_newlines += 1
                        else:
                            filtered_content += char
                            consecutive_newlines = 0
                    
                    print(filtered_content, end="", flush=True)
                    previous_length = len(current_content)
            print()
        else:
            response = self._step_with_backoff(input_message=message)

        return response

    def invoke(self, task_prompt: str):
        if self.chat_agent:
            self.num_invokes = 0
            self.stopped = False

            prompt = self.create_environment_prompt() + "\n"
            if self.first_turn:
                prompt += "You are given the following task:\n"
            else:
                prompt += "You now have the following new task:\n"
            prompt += f"<task>\n{task_prompt}\n</task>"

            logger.info(f"Invoking agent with prompt:\n{prompt}")
            response = self._step(prompt)
            if response is not None and hasattr(response, "msgs") and response.msgs:
                logger.info(f"Agent response:\n{response.msgs[0].content}")
            
            self.num_invokes += 1
            while self.toolkit is not None and not self.toolkit.completion_requested:
                logger.info("Invoking agent again as completion was not requested...")
                prompt = self.create_environment_prompt() + "\n"
                prompt += "You currently have the following task:\n" + \
                    f"<task>\n{task_prompt}\n</task>\n" + \
                    "Use your tools to perform this task, or use the end_task tool to request " + \
                    "completion of the task and to receive the next task."
                
                response = self._step(prompt)
                if response is not None and hasattr(response, "msgs") and response.msgs:
                    logger.info(f"Agent response:\n{response.msgs[0].content}")

                self.num_invokes += 1
                if self.num_invokes >= self.max_num_invokes:
                    self.toolkit.end_task(
                        status="stopped",
                        description="Stopped due to reaching maximum number of invokes",
                        summary="Stopped due to reaching maximum number of invokes",
                        )
                    self.stopped = True

            if self.toolkit:
                self.toolkit.completion_requested = False

        else:
            logger.info("No chat agent available, please provide a model")

    def create_environment_prompt(self) -> str:
        """
        Create a prompt describing the environment of the robot.

        Returns:
            str: Formatted description of the environment
        """
        environment_description = ""
        environment_description += self.env.world.get_locations_description()

        # Add current semantic location information
        current_location = self.env.world.get_current_location(self.position)
        if current_location:
            environment_description += f"\nYou are currently at location: {current_location}\n"

        return environment_description

    def move_to(self, target):
        """
        Move the robot base to a target position. Initiates movement through
        setting the target position and activating the movement state. Movement
        is handled by the on_pre_step, which is called when env.step() is called.

        Args:
            target: Either a string name of a location or [x, y] coordinates

        Returns:
            str: 'success' if movement was successful, 'failure' otherwise
        """
        current_location_name = self.env.world.get_current_location(self.position)
        logger.info(f"Robot is moving from current location: {current_location_name}; Target location: {target}")

        # Handle both location names and direct positions
        if isinstance(target, str):
            # Generate path to target location
            if current_location_name is None:
                logger.warning("Current location not found")
                return {
                    'status': 'failure',
                    'message': "Current location not found"
                }

            self.path = self.env.world.get_path_between(current_location_name, target)
            if not self.path:
                # Fallback to direct movement if no path found
                location = self.env.world.get_location(target)
                if location is None:
                    logger.warning(f"Location '{target}' not found")
                    return {
                        'status': 'failure',
                        'message': f"Location '{target}' not found"
                    }
                target_pos = location.center
                logger.info(f"No path found, moving directly to target at {target_pos}")
            else:
                logger.info(f"Path found: {[self.env.world.get_current_location(waypoint) for waypoint in self.path]}")
                # Use path planning
                self.waypoint_index = 0
                target_pos = np.array(self.path[0])
        else:
            # Direct coordinates - no path planning
            target_pos = np.array(target)

        # Set up movement state
        self.action_target = np.array([target_pos[0], target_pos[1], 0])
        self.activity.add("move")

        # Run simulation until we reach the target or timeout
        for _ in range(2000):
            if "move" not in self.activity:
                logger.info(f"Successfully reached the target {target}")
                return {
                    'status': 'success',
                    'message': f'You successfully reached the target {target}'
                }
            self.env.step(1)

        # Timeout - stop movement
        self.activity.remove("move")
        self.action_target = None

        # Reset path irrespective of whether we were following one
        self._reset_path()

        p.resetBaseVelocity(
            self.base_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )

        logger.error(f"Timed out while moving to target {target}")
        return {
            'status': 'timeout',
            'message': f'Timed out while moving to target {target}'
        }

    def handle_move(self):
        assert "move" in self.activity and self.action_target is not None
        # Need to assert that action_target is a numpy array or list of length 3

        pos = np.array(self.position)

        # Calculate direction to target
        goal_vec = self.action_target - pos
        goal_vec[2] = 0  # Keep movement in XY plane
        dist = np.linalg.norm(goal_vec)

        # Check if we've reached the target
        if dist < 0.1:
            # Check if we're following a path
            if self.path and self.waypoint_index < len(self.path) - 1:
                # Move to next waypoint
                self.waypoint_index += 1
                next_waypoint = self.path[self.waypoint_index]
                self.action_target = np.array([next_waypoint[0], next_waypoint[1], 0])
            else:
                # Reached final target
                p.resetBaseVelocity(
                    self.base_id,
                    linearVelocity=[0, 0, 0],
                    angularVelocity=[0, 0, 0]
                )
                self.activity.remove("move")
                self.action_target = None
                self._reset_path()
        else:
            # Move towards target
            v = 3 * goal_vec / dist
            p.resetBaseVelocity(
                self.base_id,
                linearVelocity=v,
                angularVelocity=[0, 0, 0]
            )

    def _reset_path(self):
        self.path = []
        self.waypoint_index = 0

    def _raise_gripper(self):
        # Move Gripper up
        pos = np.array(self.position)

        joint_angles = p.calculateInverseKinematics(self.robot_id, 6, pos + np.array([0.0, 0.0, 1.5]))  # Fix moving up
        for i, angle in enumerate(joint_angles):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=500,
                positionGain=0.03,
                velocityGain=1.0,
                maxVelocity=2.0
            )

    def grab(self, target_name_or_id) -> Dict[str, str]:
        """
        Grab an object by name or ID. Initiates grabbing through setting the
        target object and activating the grab state. Grabbing is handled by
        on_pre_step, which is called when env.step() is called.

        Args:
            target_name_or_id: Either a string name of an object or a direct object ID

        Returns:
            str: 'success' if grabbing was successful, 'failure' otherwise
        """
        logger.info(f"Robot is trying to grab item {target_name_or_id}")
        # Handle both object names and direct IDs
        if isinstance(target_name_or_id, str):
            target_id = self.env.world.get_object(target_name_or_id)
            if target_id is None:
                logger.warning(f"Object '{target_name_or_id}' not found in world objects")
                return {
                    'status': 'failure',
                    'message': f'Object "{target_name_or_id}" not found'
                }
        else:
            target_id = target_name_or_id

        # Set up grab state
        if self.held_object_id is not None:
            held_object_name = self.env.world.get_object_by_id(self.held_object_id)
            logger.info(f"Robot is already holding object {held_object_name}")
            return {
                'status': 'failure',
                'message': f'You are currently holding {held_object_name}, and cannot hold more than one item at once'
            }

        self.action_target = target_id
        self.activity.add("grab")
        logger.info(f"Grab activity initiated")

        # Run simulation until we grab the object or timeout
        for _ in range(1000):
            if "grab" not in self.activity:
                # Successfully grabbed the object
                logger.info(f"Successfully grabbed object")
                return {
                    'status': 'success',
                    'message': 'Object grabbed successfully'
                }
            self.env.step(1)

        # Timeout - failed to grab
        self.activity.remove("grab")
        self.action_target = None

        # Raise gripper after failed grab attempt
        self._raise_gripper()

        for _ in range(100):
            self.env.step(1)

        logger.error(f"Timed out trying to grab object")
        return {
            'status': 'timeout',
            'message': 'Timed out trying to grab object'
        }

    def handle_grab(self):
        assert "grab" in self.activity and self.action_target is not None
        assert type(self.action_target) is int, f"self.action_target {self.action_target} is {type(self.action_target)}, not int"

        target_pos, target_ori = p.getBasePositionAndOrientation(self.action_target)
        joint_angles = p.calculateInverseKinematics(self.robot_id, 6, target_pos)

        target_location = None

        target_object_location = self.env.world.get_location(self.env.world.get_object_location(self.action_target))
        if target_object_location:
            place_positions = target_object_location.place_positions.values()
            if place_positions:
                target_location = min(place_positions, key=lambda x: float(np.linalg.norm(np.array(x.center - np.array(target_pos)))))

        # Get end-effector position
        ee_index = 6
        # Get gripper pose
        ee_pos, ee_ori = p.getLinkState(self.robot_id, ee_index)[4:6]

        dist = np.linalg.norm(np.array(target_pos) - np.array(ee_pos))
        if dist < 0.75:
            if target_location is not None:
                target_location.occupied_by = None
                logger.info(f"Removed object {self.env.world.get_object_by_id(self.action_target)} " \
                        + f"from {target_location.name if target_location else None}")
            else: 
                logger.error(f"Target location for setting object occupancy to None was not found")

            offset = np.array([0.25, 0, 0])
            # Constraints to simulate grabbing (modified ChatGPT)
            p.resetBasePositionAndOrientation(
                self.action_target,
                np.array(ee_pos)+offset,
                target_ori
            )

            # Transform from parent world frame to parent local frame
            parent_inv_pos, parent_inv_ori = p.invertTransform(ee_pos, ee_ori)

            # Compute child pose relative to parent
            rel_pos, rel_ori = p.multiplyTransforms(
                parent_inv_pos, parent_inv_ori,
                np.array(ee_pos)+offset, target_ori
            )
            # Create constraint as grab
            self.constraint_id = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=6,
                childBodyUniqueId=self.action_target,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=rel_pos,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=rel_ori,
                childFrameOrientation=[0, 0, 0, 1]
            )

            p.changeConstraint(self.constraint_id, maxForce=500, erp=1.0)

            # Raise gripper after grab
            self._raise_gripper()

            self.held_object_id = self.action_target
            self.activity.remove("grab")
            self.action_target = None
        else:
            # Move arm towards target
            for i, angle in enumerate(joint_angles):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=angle,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1.0,
                                        maxVelocity=2.0
                )

    def place(self, location, place_position=None):
        """
        Place the currently grabbed object at a location or position. Initiates
        placement through setting the target position and activating the place state.
        Placement is handled by on_pre_step, which is called when env.step() is called.

        Args:
            location: A string name of a location
            place_position: Optional position within the location to place the object

        Returns:
            str: 'success' if placement was successful, 'failure' otherwise
        """
        logger.info(f"Robot is trying to place {self.env.world.get_object_by_id(self.held_object_id)} to {location}")
        if loc := self.env.world.get_location(location):
            target_position = None
            if place_position is None:
                target_location = loc.get_place_position(loc.get_place_positions()[0])

                if not target_location:
                    return {
                        "status": "failure",
                        "message": f"You cannot place items at the provided location {location}"
                    }
            else:
                target_location = loc.get_place_position(place_position)
                logger.info(f"Place position: {target_location.name if target_location else None}")

            if target_location is None:
                logger.error(f"The provided place position does not exist in the location {location}")
                return {
                    "status": "failure",
                    "message": f"The provided place position {place_position} does not "
                        f"exist in the location {location}"
                }
            else:
                target_position = target_location.center
                if target_location.occupied_by is not None:
                    logger.error(f"The target place position is occupied by {target_location.occupied_by}")
                    return {
                        "status": "failure",
                        "message": f"The provided place position {place_position} "
                            f"in {location} is occupied by {target_location.occupied_by}"
                    }
        else:
            return {
                "status": "failure",
                "message": f"The provided location {location} does not exist"
            }

        # Check if we're actually holding something
        if self.constraint_id is None:
            logger.error(f"The robot is not holding an object to perform a place action")
            return {
                "status": "failure",
                "message": "You are not holding anything"
            }

        # Set up place state
        self.action_target = [np.array(target_position), target_location]
        self.activity.add("place")
        logger.info(f"Place activity initiated for object at location: {target_location.name}")

        # Run simulation until we place the object or timeout
        for _ in range(1000):
            if "place" not in self.activity:
                logger.info(f"Successfully placed object at location {target_location.name}")
                return {
                    "status": "success",
                    "message": "Object placed successfully"
                }
            self.env.step(1)

        # Timeout - failed to place
        self.activity.remove("place")
        self.action_target = None

        # Raise gripper after failed place attempt
        self._raise_gripper()

        logger.error(f"Timed out trying to place object at location {location}")
        return {
            "status": "failure",
            "message": "Failed to place object"
        }

    def handle_place(self):
        assert "place" in self.activity and self.action_target is not None
        #assert isinstance(self.action_target, np.ndarray) and len(self.action_target) == 3

        placement_pos = self.action_target[0].copy()

        # Move arm to placement position
        joint_angles = p.calculateInverseKinematics(self.robot_id, 6, placement_pos)

        # Apply joint angles
        for i, angle in enumerate(joint_angles):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1.0,
                                    maxVelocity=2.0)

        # Check if we're close enough to the target position
        ee_pos, _ = p.getLinkState(self.robot_id, 6)[4:6]
        dist = np.linalg.norm(np.array(placement_pos) - np.array(ee_pos))

        if dist < 0.5:
            # Get the name of the object being placed
            object_name = self.env.world.get_object_by_id(self.held_object_id)
            self.action_target[1].occupied_by = object_name

            # Remove the constraint to release the object
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None

            # Teleport the object to the exact target position
            p.resetBasePositionAndOrientation(self.held_object_id, self.action_target[0], [0,0,0,1])

            self.held_object_id = None
            self.activity.remove("place")
            self.action_target = None

            # Raise gripper after place
            self._raise_gripper()

    def on_pre_step(self):
        if "move" in self.activity and self.action_target is not None:
            self.handle_move()
        if "grab" in self.activity and self.action_target is not None:
            self.handle_grab()
        if "place" in self.activity and self.action_target is not None:
            self.handle_place()

    def on_post_step(self):
        # Handle any post-physics updates
        # e.g., update state, check collisions
        pass
