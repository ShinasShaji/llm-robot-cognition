from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from world import World

from logging_utils import get_logger

# Initialize logger
logger = get_logger("STATE_CHECKER")


class WorldStateChecker:
    """
    A class to check world state conditions before and after robot task execution.
    """
    
    def __init__(self, world: 'World', conditions_config: Dict):
        self.world = world
        self.conditions_config = conditions_config
        self.initial_states = {}
    
    def record_initial_state(self, task_index):
        """
        Record the initial state of the world before robot invocation.
        
        Args:
            task_index: Index of the task to check conditions for
        """
        # Find the condition configuration for this task index
        task_condition = None
        for condition in self.conditions_config:
            if condition.get('task_index') == task_index:
                task_condition = condition
                break
        
        if not task_condition:
            logger.warning(f"No condition found for task index {task_index}")
            return
        
        location_info = task_condition.get('value', {}).get('location', {})
        if not location_info:
            logger.warning("No location information found in condition")
            return
        
        # Get the location name (should be shelf)
        location_name = list(location_info.keys())[0]
        place_positions = location_info[location_name].get('place_positions', [])
        
        # Record initial state for each place position
        self.initial_states[task_index] = {}
        location_obj = self.world.get_location(location_name)
        if location_obj:
            for position_name in place_positions:
                place_position = location_obj.get_place_position(position_name)
                if place_position:
                    self.initial_states[task_index][position_name] = str(place_position.occupied_by)
        
        logger.info(f"Recorded initial state for task {task_index}: {self.initial_states[task_index]}")
    
    def check_final_state(self, task_index):
        """
        Check the final state of the world after robot invocation.
        
        Args:
            task_index: Index of the task to check conditions for
            
        Returns:
            dict: Results of the condition checks
        """
        logger.info(f"Checking final state for task {task_index}...")

        results = {}
        
        # Find the condition configuration for this task index
        task_condition = None
        for condition in self.conditions_config:
            if condition.get('task_index') == task_index:
                task_condition = condition
                break
        
        if not task_condition:
            logger.error(f"No condition found for task index {task_index}")
            results['status'] = "error"
            results['error'] = f"No condition found for task index {task_index}"
            return results
        
        condition_type = task_condition.get('condition')
        targets = task_condition.get('targets', [])
        location_info = task_condition.get('value', {}).get('location', {})
        
        if not location_info:
            logger.error("No location information found in condition")
            results['status'] = "error"
            results['error'] = "No location information found in condition"
            return results
        
        # Get the location name (should be shelf)
        location_name = list(location_info.keys())[0]
        place_positions = location_info[location_name].get('place_positions', [])
        
        # Get the location object from the world
        location_obj = self.world.get_location(location_name)
        if not location_obj:
            logger.error(f"Location '{location_name}' not found in world")
            results['status'] = "error"
            results['error'] = f"Location '{location_name}' not found in world"
            return results
        
        if condition_type == "occupied_by":
            # Check if the specified place positions are occupied by the target objects
            occupied_results = {}
            
            num_success = 0
            for position_name in place_positions:
                place_position = location_obj.get_place_position(position_name)
                if not place_position:
                    occupied_results[position_name] = {
                        'status': 'error',
                        'message': f"Place position '{position_name}' not found"
                    }
                    continue
                
                occupied_by = place_position.occupied_by
                if occupied_by in targets:
                    occupied_results[position_name] = {
                        'status': 'success',
                        'message': f"Position '{position_name}' correctly occupied by '{occupied_by}'"
                    }
                    num_success += 1
                elif occupied_by is None:
                    occupied_results[position_name] = {
                        'status': 'failure',
                        'message': f"Position '{position_name}' is empty, expected one of {targets}"
                    }
                else:
                    occupied_results[position_name] = {
                        'status': 'failure',
                        'message': f"Position '{position_name}' occupied by '{occupied_by}', expected one of {targets}"
                    }
            
            results['occupied_by'] = occupied_results
            if num_success == len(targets):
                results["status"] = "success"
            else:
                results['status'] = "failure"
            
        elif condition_type == "swap":
            # For swap condition, check if targets have swapped positions compared to initial state
            swap_results = {}
            
            initial_state = self.initial_states.get(task_index, {})
            
            num_success = 0
            for position_name in place_positions:
                place_position = location_obj.get_place_position(position_name)
                if not place_position:
                    swap_results[position_name] = {
                        'status': 'error',
                        'message': f"Place position '{position_name}' not found"
                    }
                    continue
                
                current_occupied_by = place_position.occupied_by
                initial_occupied_by = initial_state.get(position_name)
                
                # Check if there was a swap between target objects
                if (current_occupied_by in targets and
                    initial_occupied_by in targets and
                    current_occupied_by != initial_occupied_by):
                    swap_results[position_name] = {
                        'status': 'success',
                        'message': f"Position '{position_name}' swapped from '{initial_occupied_by}' to '{current_occupied_by}'"
                    }
                    num_success += 1
                elif current_occupied_by == initial_occupied_by:
                    swap_results[position_name] = {
                        'status': 'info',
                        'message': f"Position '{position_name}' unchanged from initial state ('{initial_occupied_by}')"
                    }
                elif current_occupied_by in targets:
                    swap_results[position_name] = {
                        'status': 'info',
                        'message': f"Position '{position_name}' now occupied by '{current_occupied_by}' (target object)"
                    }
                elif initial_occupied_by in targets:
                    swap_results[position_name] = {
                        'status': 'info',
                        'message': f"Position '{position_name}' previously occupied by '{initial_occupied_by}' (target object) is now empty or occupied by non-target"
                    }
                else:
                    swap_results[position_name] = {
                        'status': 'info',
                        'message': f"Position '{position_name}' unchanged (not involving target objects)"
                    }
            
            results['swap'] = swap_results
            if num_success == len(targets):
                results["status"] = "success"
            else:
                results['status'] = "failure"
        
        return results