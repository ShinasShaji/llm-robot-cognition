from typing import List, Optional, TYPE_CHECKING, Union
from functools import wraps
from camel.toolkits import FunctionTool, BaseToolkit

if TYPE_CHECKING:
    from robot import Robot
    from memory import Memory

from logging_utils import get_logger

# Initialize logger
logger = get_logger("TOOLKIT")

class RobotToolkit(BaseToolkit):
    """Toolkit for robot actions like moving, grabbing, placing objects, and memory operations."""

    def __init__(self, robot: 'Robot', memory: 'Memory'):
        """Initialize the toolkit with robot instance and memory.

        Args:
            robot: The Robot instance that will execute the actions
            memory: The Memory instance for storing and retrieving memories
        """
        self.robot = robot
        self.memory = memory
        self.scratchpad = []  # List to store scratchpad entries
        
        # Track the number of tool calls that occured
        self.num_toolcalls = 0

        # Track the status of the result from end_task
        self.end_task_status = None

        # Flag to detect that the model has called the summarize_task tool
        self.completion_requested = False

        super().__init__()

    @staticmethod
    def tool_call_counter(func):
        """Decorator to increment num_toolcalls counter."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.num_toolcalls += 1
            return func(self, *args, **kwargs)
        return wrapper

    @tool_call_counter
    def look_around(self) -> dict:
        """Look around and return information about the environment. This method
        creates an environment prompt that describes the current state of the
        world, including locations and objects.

        Returns:
            dict: Result containing status and message with environment description.
        """
        content = self.robot.create_environment_prompt()
        logger.info(f"Robot looked around to find: \n{content}")

        result = {
            "status": "success",
            "message": content
        }

        return result

    @tool_call_counter
    def move_to(self, target: str) -> dict:
        """Move to a target location.

        Args:
            target: Either a string name of a location or [x, y] coordinates.

        Returns:
            dict: Result of the movement action including status.
        """
        result = self.robot.move_to(target)
        return result

    @tool_call_counter
    def grab(self, target: str) -> dict:
        """Grab an object by name or ID. You must move to the location containing
        the object first before grabbing it. Grabbing an object makes the object
        the currently held object.

        Args:
            target: Either a string name of an object or a direct object ID.

        Returns:
            dict: Result of the grab action including status.
        """
        result = self.robot.grab(target)
        return result

    @tool_call_counter
    def place(self, location: str, place_position: str) -> dict:
        """Place the currently held object at a target location. You must move to
        the location first before placing the object there, and must also have an
        object that is currently being held.

        Args:
            location: A string name of a location
            place_position: Place position within the location to place the object

        Returns:
            dict: Result of the place action including status.
        """
        result = self.robot.place(location, place_position)
        return result

    @tool_call_counter
    def end_task(self, status: str, description: str, summary: str) -> dict:
        """End the current task with a status report.

        Args:
            status: Whether the task execution was a success, a failure, or unknown.
            description: A description of the task, that is independent of what was actually done.
                This is a description of the task given to the robot.
            summary: A description of the execution trace of the robot to perform the task. Be
                sure to include any relevant details about the task execution in detail.

        Returns:
            dict: Result confirming the task completion.
        """
        result = {
            "status": status,
            "description": description,
            "summary": summary
        }
        logger.info(f"Agent has ended task with result:\n")
        logger.info(f"status: {result['status']}")
        logger.info(f"description: {result['description']}")
        logger.info(f"summary: {result['summary']}")
        
        # Store the task status for evaluation
        self.end_task_status = status
        
        # Store the task result in memory
        metadata = {"status": status, "summary": summary}

        if not status == "stopped":
            self.memory.add_memory(description, metadata)

        self.completion_requested = True
        
        return result

    @tool_call_counter
    def search_memory(self, query: str) -> dict:
        """Search the robot's memory for relevant past tasks.

        Args:
            query: A query string to search for in the memory

        Returns:
            dict: Search results with status and list of matching memories
        """
        logger.info(f"Robot is querying memories with the query: {query}")
        # Search for memories using the query
        memories = self.memory.search_memories(query)
        
        # Format the results to be similar to finish_task output
        formatted_results = []
        for memory in memories:
            formatted_result = {
                "status": memory["metadata"].get("status", "unknown"),
                "description": memory["content"],
                "summary": memory["metadata"].get("summary", "")
            }
            formatted_results.append(formatted_result)

        if formatted_results:
            logger.info("The query returned the following results:")
            for formatted_result in formatted_results:
                logger.info(f"status: {formatted_result['status']}")
                logger.info(f"description: {formatted_result['description']}")
                logger.info(f"summary: {formatted_result['summary']}")
                logger.info("---")
            logger.info("\n")
        else:
            logger.info("The memory contained no matching results.")
            formatted_results = "The memory contained no matching results."
        
        return {
            "status": "success",
            "query": query,
            "results": formatted_results
        }

    @tool_call_counter
    def add_to_scratchpad(self, content: str) -> dict:
        """Add an entry to the scratchpad for reasoning and reflection.
        
        Args:
            content (str): A string entry with content to add to the scratchpad
            
        Returns:
            dict: Confirmation of the addition with status
        """
        self.scratchpad.append(content)
        logger.info(f"The agent wrote to the scratchpad: \n{content}\n")
        return {
            "status": "success",
            "message": f"Added entry to scratchpad. Current scratchpad has {len(self.scratchpad)} entries."
        }

    @tool_call_counter
    def view_scratchpad(self) -> dict:
        """View the current scratchpad contents joined as paragraphs.
        
        Returns:
            dict: The scratchpad contents with status
        """
        if not self.scratchpad:
            return {
                "status": "success",
                "message": "Scratchpad is empty.",
                "content": ""
            }
            
        # Join entries into paragraphs (separated by blank lines)
        content = "\n\n".join(self.scratchpad)
        logger.info(f"The agent viewed the scratchpad containing: \n{content}\n")
        return {
            "status": "success",
            "message": f"Scratchpad contains {len(self.scratchpad)} entries.",
            "content": content
        }

    def get_tools(self) -> list[FunctionTool]:
        """Get list of available tools."""
        return [
            FunctionTool(self.look_around, ),
            FunctionTool(self.move_to),
            FunctionTool(self.grab),
            FunctionTool(self.place),
            FunctionTool(self.end_task),
            FunctionTool(self.search_memory),
            FunctionTool(self.add_to_scratchpad),
            FunctionTool(self.view_scratchpad)
        ]
