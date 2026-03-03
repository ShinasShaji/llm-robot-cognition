import numpy as np
import pybullet as p

from typing import Dict, List, Optional, Union

from logging_utils import get_logger

# Initialize logger
logger = get_logger("WORLD")

Mug_scale = 0.1
Table_scale = 0.5
TV_scale = 0.8
Shelf_scale = 0.25

class Location():
    def __init__(self,
                 name: str,
                 center: List[float],
                 place_position: Optional[Union[Dict[str, List[float]], bool]] = None
            ):
        self.name = name
        self.center = np.array(center)
        self.place_positions: Dict[str, 'Location'] = {}

        # Add defined place positions, if any
        if isinstance(place_position, dict):
            self.add_place_positions(place_position)
        self.neighbours: List['Location']= []

        self.occupied_by: Optional[str] = None  # Store the name of the object placed here, or None if empty

    def next_to(self, neighbours: List['Location']) -> None:
        for neighbour in neighbours:
            if neighbour not in self.neighbours:
                self.neighbours.append(neighbour)
                neighbour.next_to([self])

    def get_place_position(self, name: str) -> Optional['Location']:
        return self.place_positions.get(name.lower().replace(" ", "_"))
    
    def get_place_positions(self) -> List[str]:
        return list(self.place_positions.keys())

    def add_place_positions(self, positions: Dict[str, List[float]]) -> None:
        for name, position in positions.items():
            self.place_positions[name] = Location(name, position, place_position=False)

class PlannerNode():
    def __init__(self, location: 'Location'):
        self.location = location
        self.f = 0
        self.g = 0
        self.path = []


class World():
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.objects: Dict[str, int] = {}

    def add_location(self, location: Location) -> None:
        self.locations[location.name.lower().replace(" ", "_")] = location

    def add_next_to(self, location: str, neighbours: List[str]) -> None:
        self.locations[location.lower().replace(" ", "_")].next_to(
            [self.locations[neighbour.lower().replace(" ", "_")]
                for neighbour in neighbours])
        for neighbour in neighbours:
            neighbour_location = self.locations[neighbour.lower().replace(" ", "_")]
            neighbour_location.next_to([self.locations[location.lower().replace(" ", "_")]])

    def get_location(self, name: str) -> Optional[Location]:
        return self.locations.get(name.lower().replace(" ", "_"))

    def add_object(self, object_id: int, name: str) -> None:
        self.objects[name.lower()] = object_id

    def get_object(self, name: str) -> Optional[int]:
        return self.objects.get(name.lower())

    def get_object_by_id(self, object_id: int) -> Optional[str]:
        object_names = [obj_name for obj_name, obj_id in self.objects.items() if object_id == obj_id]
        if object_names:
            return object_names[0]

        return None

    def get_current_location(self, position):
        """Get the current location of the robot."""
        position = np.array(position)[:2]  # Only consider x,y coordinates

        # Get array of all location centers
        centers = np.array([loc.center for loc in self.locations.values()])

        # Calculate distances to all locations at once
        distances = np.linalg.norm(centers - position, axis=1)

        # Get index of minimum distance
        min_index = np.argmin(distances)

        # Return the location name
        return list(self.locations.keys())[min_index]

    def get_object_location(self, object_id: int) -> Optional[str]:
        """Get the location name of an object based on its position."""
        try:
            pos, _ = p.getBasePositionAndOrientation(object_id)
            return self.get_current_location(pos)
        except ValueError:
            logger.error(f"Object of id {object_id} could not be found in the world")
            raise ValueError(f"Object of id {object_id} could not be found in the world")

    def get_objects_by_location(self) -> Dict[str, List[str]]:
        """Get a mapping of location names to lists of object names in those locations."""
        location_objects = {loc_name: [] for loc_name in self.locations.keys()}

        for obj_name, obj_id in self.objects.items():
            obj_location = self.get_object_location(obj_id)
            if obj_location and obj_location in location_objects:
                location_objects[obj_location].append(obj_name)

        return location_objects

    def get_path_between(self, start_name: str, end_name: str) -> List[List[float]]:
        start = self.get_location(start_name)
        end = self.get_location(end_name)

        if start is None or end is None:
            logger.warning("Invalid start or end")
            return []

        goal = False
        checked = [start]
        frontier = []

        current_Node = PlannerNode(start)
        frontier.append(current_Node)

        while frontier != [] and not goal:
            frontier.sort(key=lambda x: x.f)
            current_Node = frontier.pop(0)

            #print(current_Node.location.center)

            for neigh in current_Node.location.neighbours:
                if neigh not in checked:
                    # Goal check
                    if neigh == end:
                        goal = True
                        current_Node.path.append(end.center)
                        return current_Node.path
                    checked.append(neigh)

                    # Add new location to frontier and calcualte its f value
                    next_Node = PlannerNode(neigh)
                    next_Node.path = current_Node.path.copy()
                    next_Node.path.append(neigh.center)
                    next_Node.g = current_Node.g + np.linalg.norm(next_Node.location.center - current_Node.location.center)
                    next_Node.f = next_Node.g + np.linalg.norm(end.center - neigh.center)

                    frontier.append(next_Node)

        return []

    def get_locations_description(self) -> str:
        """
        Generate a string description of all locations in the world and their neighbours.

        Returns:
            str: Formatted description of locations and their neighbours
        """
        if not self.locations:
            return "No locations exist in the world."

        description = "The following locations exist in the world:\n"

        # Get objects by location for inclusion in descriptions
        location_objects = self.get_objects_by_location()

        for location_name, location in self.locations.items():
            description += f"- {location.name}\n"
            # Add objects in this location
            objects_in_location = location_objects.get(location_name, [])
            if objects_in_location:
                description += "  - Objects in location:\n"
                for obj_name in objects_in_location:
                    description += f"    - {obj_name}\n"

            place_positions_in_location = location.place_positions.keys()
            if place_positions_in_location:
                description += "  - Place positions in location:\n"
                for place_name, place_position in location.place_positions.items():
                    if place_position.occupied_by is not None:
                        description += f"    - {place_name}: occupied by {place_position.occupied_by}\n"
                    else:
                        description += f"    - {place_name}: empty\n"

            if location.neighbours:
                description += "  - Neighbouring Locations:\n"
                for neighbour in location.neighbours:
                    description += f"    - {neighbour.name}\n"

        return description

    @classmethod
    def create_default_world(cls):
        ## TODO: Either make the location determine where the model, like the shelf, spawn or the other way around
        # Currently you can tell the robot the shelf is to the right, while the model is 500m in the air

        world = cls()

        ### Create locations using the new Location class

        ## Hallway
        locations = [
            Location("Hallway Area Door", [3,0],{
                "floor": [4,0,0.2]
            }),
            Location("Front Door", [0,0], {
                "floor": [-1,0,0.2]
            }),
            Location("Kitchen Door", [3,2]),
            Location("Living Room Door", [3,-2]),
            Location("Living Room TV", [8.0, -4.5]),
            Location("Living Room", [4.0, -4.0], {
                "floor": [3,-4,0.2]
            }),

            ## Kitchen
            Location("Kitchen Area Left", [3, 4], {
                "floor": [3,5,0.2]
            }
            ),
            Location("Kitchen Shelf", [1, 4.85], {
                "top": [1,6,4.5*Shelf_scale+0.1],
                "middle": [1,6,3.0*Shelf_scale+0.1],
                "bottom": [1,6,1.5*Shelf_scale+0.1]
            }),
            Location("Kitchen Table", [6.0, 5.0], {
                "middle": [6.0, 6.0, 2.0*Table_scale+0.1]
            }),
        ]

        # Add locations to world
        for loc in locations:
            world.add_location(loc)

        table_location = world.get_location("Kitchen Table")
        if table_location:
            table_place_location = table_location.get_place_position("middle")
            if table_place_location is not None:
                table_place_location.occupied_by = "mug"

        ## Set up location relationships using the world's add_next_to method
        # Hallway
        world.add_next_to("Hallway Area Door", ["Front Door", "Kitchen Door", "Living Room Door"])

        # Front door
        # world.add_next_to("Front Door", ["Kitchen Door, Living Room Door"])

        # Kitchen
        world.add_next_to("Kitchen Area Left", ["Kitchen Door", "Kitchen Shelf", "Kitchen Table"])

        # Living room
        world.add_next_to("Living Room", ["Living Room Door", "Living Room TV"])


        return world

    def create_default_physical_objects(self):
        """Create physical objects in the PyBullet simulation.
        This should be called after PyBullet has been initialized."""

        # Visual to make collision boxes invisible
        empty_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1,1,1],
            rgbaColor=[0, 0, 0, 0]     # fully transparent
        )


        # Create test cube
        half_size = [0.1, 0.1, 0.1]
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size)
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        cube_id = p.createMultiBody(baseMass=0.01,
                                   baseCollisionShapeIndex=cube_collision,
                                   baseVisualShapeIndex=cube_visual,
                                   basePosition=[-1.0, 0.0, 0.1])

        # Add cube to world objects
        self.add_object(cube_id, "cube")

        # Create box cube at Living Room Door
        half_size = [0.1, 0.1, 0.1]
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size)
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        box_id = p.createMultiBody(baseMass=0.01,
                                   baseCollisionShapeIndex=box_collision,
                                   baseVisualShapeIndex=box_visual,
                                   basePosition=[3,-4,0.1])

        # Add box to world objects
        self.add_object(box_id, "box")


        # Create Mug
        Mug_pos = np.array([6.0, 6.0, 2.0*Table_scale+0.1])
        Mug_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0])

        half_size = np.array([1.4, 1, 1.0])*Mug_scale
        Mug_visual = p.createVisualShape(p.GEOM_MESH, fileName="./models/Mug.obj", meshScale=[Mug_scale, Mug_scale, Mug_scale])
        Mug_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        Mug_id = p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=Mug_visual,
            basePosition=Mug_pos,
            baseOrientation=Mug_orientation,      # Apply the rotation here

            # Add links for each collision box
            linkMasses=[0.01],
            linkCollisionShapeIndices=[Mug_collision],
            linkVisualShapeIndices=[empty_visual],
            linkPositions=[[0.0,0.0,half_size[2]]],
            linkOrientations=[[0,0,0,1]],
            linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]]
        )
        # Add TV to world objects
        self.add_object(Mug_id, "mug")


        # Create TV
        TV_pos = np.array([8.0, -6.0, 0.0])
        TV_orientation = p.getQuaternionFromEuler([0.0, 0.0, np.pi])

        half_size = np.array([1.35, 1, 1.3])*TV_scale
        TV_visual = p.createVisualShape(p.GEOM_MESH, fileName="./models/TV.obj", meshScale=[TV_scale, TV_scale, TV_scale])
        TV_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        TV_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=TV_visual,
            basePosition=TV_pos,
            baseOrientation=TV_orientation,      # Apply the rotation here

            # Add links for each collision box
            linkMasses=[0],
            linkCollisionShapeIndices=[TV_collision],
            linkVisualShapeIndices=[empty_visual],
            linkPositions=[[0.0,0.0,half_size[2]]],
            linkOrientations=[[0,0,0,1]],
            linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]]
        )
        # Add TV to world objects
        self.add_object(TV_id, "tv")

        # Create Table
        Table_pos = np.array([6.0, 6.0, 0.0])
        Table_orientation = p.getQuaternionFromEuler([0.0, 0.0, np.pi/2])

        half_size = np.array([1.0, 1.5, 1.0])*Table_scale
        Table_visual = p.createVisualShape(p.GEOM_MESH, fileName="./models/Table.obj", meshScale=[Table_scale, Table_scale, Table_scale])
        Table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
        Table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=Table_visual,
            basePosition=Table_pos,
            baseOrientation=Table_orientation,      # Apply the rotation here

            # Add links for each collision box
            linkMasses=[0],
            linkCollisionShapeIndices=[Table_collision],
            linkVisualShapeIndices=[empty_visual],
            linkPositions=[[0.0,0.0,half_size[2]]],
            linkOrientations=[[0,0,0,1]],
            linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]]
        )
        # Add TV to world objects
        self.add_object(Table_id, "table")

        # Define the location and rotation for the entire shelf
        shelf_location = np.array([1.0, 6.0, 0.0])

        shelf_orientation = p.getQuaternionFromEuler([0, 0, np.pi])

        # Collision shapes for the simplified planes (links)
        # Define half-extents for plates and supports
        plate_dims = np.array([3.0, 2.0, 0.02])*Shelf_scale
        support_dims_side = np.array([1/3, 2.0, 6.0])*Shelf_scale
        support_dims_back = np.array([3.0, 1/3, 6.0])*Shelf_scale


        plate_pos = [
            [0, 0, 1.5*Shelf_scale],  # Bottom shelf
            [0, 0, 3.0*Shelf_scale],  # Middle shelf
            [0, 0, 4.5*Shelf_scale]   # Top shelf
        ]
        support_pos = [
            np.array([-1.5, 0, 3.0])*Shelf_scale,   # Left support
            np.array([1.5, 0, 3.0])*Shelf_scale,    # Right support
            np.array([0, -5/6, 3.0])*Shelf_scale    # Back support
        ]

        # Create collision shapes for the plates and supports
        collision_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in plate_dims])
        collision_support_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in support_dims_side])
        collision_support_back = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in support_dims_back])


        # Create the visual shape from your Blender mesh
        visual_shelf_id = p.createVisualShape(p.GEOM_MESH, fileName="./models/Shelf.obj", meshScale=[Shelf_scale, Shelf_scale, Shelf_scale])

        # Create the multibody object
        # This is where we apply the location and orientation to the base
        shelf_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shelf_id,
            basePosition=shelf_location,            # Correctly position the entire shelf
            baseOrientation=shelf_orientation,      # Apply the rotation here

            # Add links for each collision box
            linkMasses=[0] * 6,
            linkCollisionShapeIndices=[
                collision_box_id, collision_box_id, collision_box_id,
                collision_support_side, collision_support_side, collision_support_back
            ],
            linkVisualShapeIndices=[empty_visual] * 6,
            linkPositions=plate_pos + support_pos,
            linkOrientations=[[0,0,0,1]] * 6,
            linkInertialFramePositions=[[0,0,0]] * 6,
            linkInertialFrameOrientations=[[0,0,0,1]] * 6,
            linkParentIndices=[0] * 6,
            linkJointTypes=[p.JOINT_FIXED] * 6,
            linkJointAxis=[[0, 0, 0]] * 6
        )
        self.add_object(shelf_body, "shelf")


        # Define the location and rotation for the walls
        walls_location = np.array([0.0, 0.0, 0.0])
        walls_scale = 1.0

        walls_orientation = p.getQuaternionFromEuler([0, 0, -np.pi/2])

        wall_dims = [
            np.array([6.3, 0.3, 3.0])*walls_scale,
            np.array([6.3, 0.3, 3.0])*walls_scale,

            np.array([0.3, 10.3, 3.0])*walls_scale,
            np.array([0.3, 10.3, 3.0])*walls_scale,

            np.array([0.3, 6.3, 3.0])*walls_scale,
            np.array([0.3, 6.3, 3.0])*walls_scale,

            np.array([0.3, 1.7, 3.0])*walls_scale,
            np.array([0.3, 1.7, 3.0])*walls_scale,

            np.array([5.6, 0.3, 3.0])*walls_scale,
            np.array([5.6, 0.3, 3.0])*walls_scale
        ]

        walls_pos = [
            np.array([-4.15, 0.15, 1.5])*walls_scale,
            np.array([4.15, 0.15, 1.5])*walls_scale,

            np.array([7.15, 5.15, 1.5])*walls_scale,
            np.array([-7.15, 5.15, 1.5])*walls_scale,

            np.array([1.85, 7.15, 1.5])*walls_scale,
            np.array([-1.85, 7.15, 1.5])*walls_scale,

            np.array([1.85, 1.15, 1.5])*walls_scale,
            np.array([-1.85, 1.15, 1.5])*walls_scale,

            np.array([4.5, 10.15, 1.5])*walls_scale,
            np.array([-4.5, 10.15, 1.5])*walls_scale
        ]

        # Create collision shapes for the plates and supports
        wall_collision_boxes = []

        for wall_dim in wall_dims:
            wall_collision_boxes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in wall_dim]))

        # Create the visual shape from your Blender mesh
        visual_walls_id = p.createVisualShape(p.GEOM_MESH, fileName="./models/Walls.obj", meshScale=[walls_scale, walls_scale, walls_scale])

        # Create the multibody object
        # This is where we apply the location and orientation to the base
        wall_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_walls_id,
            basePosition=walls_location,
            baseOrientation=walls_orientation,      # Apply the rotation here

            # Add links for each collision box
            linkMasses=[0] * 10,
            linkCollisionShapeIndices=wall_collision_boxes,
            linkVisualShapeIndices=[empty_visual] * 10,
            linkPositions=walls_pos,
            linkOrientations=[[0,0,0,1]] * 10,
            linkInertialFramePositions=[[0,0,0]] * 10,
            linkInertialFrameOrientations=[[0,0,0,1]] * 10,
            linkParentIndices=[0] * 10,
            linkJointTypes=[p.JOINT_FIXED] * 10,
            linkJointAxis=[[0, 0, 0]] * 10
        )
        self.add_object(wall_body, "walls")

        return self.objects
