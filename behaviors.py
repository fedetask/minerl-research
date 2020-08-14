"""
This module defines the behaviors.

Some useful conventions:
    Coordinates: World coordinates are defined as numpy arrays of floats, in the form [x, z, y].
                 See the Malmo tutorial for the orientation of axis and angle convention.
                 https://microsoft.github.io/malmo/0.17.0/Python_Examples/Tutorial.pdf

    Blocks:      Block coordinates are expressed in the Minecraft convention, i.e. the
                 coordinates of a block are floor(p) with p being any point inside the block.

                 When dealing with the GRID observation, we always represent a block as a tuple
                 (x, z, y). Occasionally, this may be converted inside functions to a
                 numpy.ndarray of int type for more efficient computation, but must always be
                 converted back to a tuple before returning.

                When expressing blocks in the world frame, they are represented by a numpy array of
                floats, always in the Minecraft convention. Therefore, any operation such as destroy
                or move to should take care of adding [0.5, 0.5, 0.5] to the block coordinates in
                order to target the center of the block.
"""


from py_trees.behaviour import Behaviour
from py_trees.blackboard import Client
from py_trees.common import Status
from blackboards import Namespace, read_command, set_command, set_action, read_observation
from blackboards import register_actions, register_commands, register_observations

import numpy as np
import itertools as it
import heapq
from collections import deque
import time

from Keys import Observations, Actions, Commands, Items
from malmo_env import MalmoEnv
from crafting import can_craft, get_tool_to_destroy


# --------------------------------- Behaviors ----------------------------------------------

class MoveTo(Behaviour):
    """Behavior that moves the player towards the destination specified in the commands blackboard

    """
    # Maximum distance allowed to the target path point, over which the path is recomputed
    RECOMPUTE_PATH_THRESHOLD = 3

    # Tolerance used by the move_to() action to check wheter we arrived at destination
    PATH_POINT_TOLERANCE = np.array([.5, .5, .1])

    # Tolerance used in looking for the next point in the path
    NEXT_POINT_MOVE_SEARCH_TOLERANCE = np.array([1.5, 1.5, 1.5])
    NEXT_POINT_MOVE_DESTROY_TOLERANCE = np.array([1, 1, 3])

    def __init__(self, name, destination_key, tolerance):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.destination_key = destination_key
        self.tolerance = tolerance
        self.path_destination = None
        self.path = None

    def setup(self):
        super().setup()
        register_actions(client=self.blackboard, can_write=True)
        register_commands(client=self.blackboard, can_write=False)
        register_observations(client=self.blackboard, can_write=False)

    def initialise(self):
        super().initialise()
        dest = read_command(self.blackboard, self.destination_key)
        grid = read_observation(self.blackboard, Observations.GRID)
        cur_pos = get_current_pos(self.blackboard)
        # If first initialization or destination changed, or next point too far:
        if self.path is None:
            recompute_path = True
        else:
            destination_changed = (dest - self.path_destination).any()
            next_too_far = False
            if len(self.path) > 0:
                next_idx = self._get_next_point(cur_pos,
                                                MoveTo.NEXT_POINT_MOVE_SEARCH_TOLERANCE,
                                                MoveTo.NEXT_POINT_MOVE_DESTROY_TOLERANCE)
                next_too_far = next_idx is None
            recompute_path = destination_changed or next_too_far

        if recompute_path:
            dest_block = get_grid_block(self.blackboard, dest)
            self.path = compute_path_astar(cur_pos, dest_block, grid, cost_for_move,
                                           cost_for_destroy, self.tolerance)
            self.path_destination = dest

    def terminate(self, new_status):
        super().terminate(new_status)
        set_action(self.blackboard, Actions.MOVE, 0)
        set_action(self.blackboard, Actions.JUMP, 0)
        set_action(self.blackboard, Actions.PITCH, 0)
        set_action(self.blackboard, Actions.TURN, 0)

    def update(self):
        destination = read_command(self.blackboard, self.destination_key)

        if self.path is None:
            # this can happen when we target a falling floating object mid-air.
            return Status.FAILURE
        offset = np.array([.5, .5, 0])  # Offset for x,z center of block

        cur_pos = get_current_pos(self.blackboard)
        if within_tolerance_arrays(a=cur_pos, b=destination + offset, tolerance=self.tolerance):
            return Status.SUCCESS

        if len(self.path) == 0:  # We have not reached the destination but path is empty :(
            return Status.FAILURE  # TODO: Understand why this happens

        next_point_idx = self._get_next_point(cur_pos,
                                              MoveTo.NEXT_POINT_MOVE_SEARCH_TOLERANCE,
                                              MoveTo.NEXT_POINT_MOVE_DESTROY_TOLERANCE)
        if next_point_idx is None:
            return Status.RUNNING  # We moved outside the path, that will be therefore recomputed

        op, next_point = self.path[next_point_idx]
        next_point_center = next_point + offset
        if op == 'MOVE_TO':
            move_status = move_to(self.blackboard, cur_pos,
                                  next_point_center,
                                  MoveTo.PATH_POINT_TOLERANCE)
            if move_status == Status.FAILURE:
                return Status.FAILURE
        elif op == 'DESTROY_BLOCK':
            destroy_status = destroy_block(self.blackboard, next_point)
            if destroy_status == Status.FAILURE:
                return Status.FAILURE
        return Status.RUNNING

    def _get_next_point(self, cur_pos, tolerance_move, tolerance_destroy):
        """Compute the next path point by taking the point within the given tolerance that is
        further down the path.

        Args:
            cur_pos (np.ndarray):           Current position in world frame.
            tolerance_move (np.ndarray):    Tolerance for selecting candidate move to points.
            tolerance_destroy (np.ndarray): Tolerance for selecting candidate destroy block points.

        Returns:
            The index of the next point in the path.

        """
        offset = np.array([.5, .5, 0])
        indices = []
        for idx in range(len(self.path)):
            op, next_point = self.path[idx]
            if op == 'MOVE_TO':
                if within_tolerance_arrays(cur_pos, next_point + offset, tolerance_move):
                    if idx - 1 < 0:
                        indices.append(idx)
                    elif self.path[idx - 1][0] != 'DESTROY_BLOCK':
                        indices.append(idx)
            elif op == 'DESTROY_BLOCK':
                if within_tolerance_arrays(cur_pos, next_point + offset, tolerance_destroy):
                    indices.append(idx)

        if len(indices) == 0:
            return None
        else:
            return np.max(indices)


class IsCloseToBlock(Behaviour):

    def __init__(self, name, item, tolerance, out_variable):
        """Behavior that checks whether the player is close to an item within the given tolerance.

        Args:
            name (str): Unique name for the behavior
            item (str): The key of the item, must belong to Keys.Items
            tolerance (list, None): List of three values corresponding to the tolerance in the x, z,
                y axes. If tolerance is None for one or more dimensions, the whole grid along those
                dimensions is evaluated.
            out_variable (str): Variable in which to write the position of the item, if close
                within the given tolerance. If None, nothing will be written to the blackboard.
        """
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.target_item = item
        self.tol = tolerance
        self.out_variable = out_variable
        self.x_lim = self.y_lim = self.z_lim = None

    def setup(self):
        register_commands(client=self.blackboard, can_write=True)
        register_observations(client=self.blackboard, can_write=False)
        grid_dim = read_observation(self.blackboard, Observations.GRID_DIM)
        grid_center = get_grid_center(grid_dim)
        if self.tol is None or self.tol[0] is None:
            self.x_lim = [0, grid_dim['x_size']]
        else:
            self.x_lim = [grid_center[0] - self.tol[0], grid_center[0] + self.tol[0] + 1]
        if self.tol is None or self.tol[1] is None:
            self.z_lim = [0, grid_dim['z_size']]
        else:
            self.z_lim = [grid_center[1] - self.tol[1], grid_center[1] + self.tol[1] + 1]
        if self.tol is None or self.tol[2] is None:
            self.y_lim = [0, grid_dim['y_size']]
        else:
            self.y_lim = [grid_center[2] - self.tol[2], grid_center[2] + self.tol[2] + 1]

    def update(self):
        grid = read_observation(self.blackboard, Observations.GRID)
        # Take subgrid around the center with range given by self.tol
        sub_grid = grid[self.x_lim[0]: self.x_lim[1],
                        self.z_lim[0]: self.z_lim[1],
                        self.y_lim[0]: self.y_lim[1]]

        # We save indices of where the subgrid contains the target item to efficiently compute
        # the distances from the center, and writing the closest one in world coordinates
        sub_grid_center = (np.array(sub_grid.shape) - 1) / 2
        coords_mat = np.array(np.where(sub_grid == self.target_item))
        if coords_mat.shape[-1] == 0:
            return Status.FAILURE
        if self.out_variable is not None:
            squared_dist = np.sum((coords_mat - sub_grid_center[:, None])**2, axis=0)
            closest_block = coords_mat[:, np.argmin(squared_dist)]
            closest_block_relative = closest_block - sub_grid_center
            cur_player_pos = get_current_pos(self.blackboard)
            target_pos = np.floor(cur_player_pos + closest_block_relative)
            set_command(self.blackboard, self.out_variable, target_pos)
        return Status.SUCCESS


class HasItem(Behaviour):

    def __init__(self, name, item, quantity=None):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name)
        self.item = item
        self.quantity = quantity

    def setup(self, **kwargs):
        super().setup(**kwargs)
        register_observations(self.blackboard, can_write=False)

    def update(self):
        if has_item(self.blackboard, self.item, self.quantity):
            return Status.SUCCESS
        else:
            return Status.FAILURE


class Craft(Behaviour):

    def __init__(self, name, item):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name)
        self.item = item

    def setup(self, **kwargs):
        super().setup(**kwargs)
        register_observations(self.blackboard, can_write=False)
        register_actions(self.blackboard, can_write=True)

    def update(self):
        inventory = read_observation(self.blackboard, Observations.INVENTORY)
        if can_craft(self.item, inventory):
            set_action(self.blackboard, Actions.CRAFT, self.item)
            return Status.RUNNING
        else:
            return Status.FAILURE


class Destroy(Behaviour):

    def __init__(self, name, comm_variable):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client(name)
        self.comm_variable = comm_variable
        self.target_coords = None

    def setup(self, **kwargs):
        super().setup(**kwargs)
        register_observations(self.blackboard, can_write=False)
        register_actions(self.blackboard, can_write=True)
        register_commands(self.blackboard, can_write=False)

    def initialise(self):
        super().initialise()
        self.target_coords = read_command(self.blackboard, self.comm_variable)

    def terminate(self, new_status):
        super().terminate(new_status)
        self.target_coords = None
        set_action(self.blackboard, Actions.ATTACK, 0)

    def update(self):
        return destroy_block(self.blackboard, self.target_coords)


class IsCloseToEntity(Behaviour):

    def __init__(self, name, entity, tolerance=None, out_variable=None):
        super().__init__(name)
        self.entity = entity
        self.tolerance = tolerance
        self.comm_variable = out_variable
        self.blackboard = self.attach_blackboard_client(name=self.name)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        register_observations(self.blackboard, can_write=False)
        register_commands(self.blackboard, can_write=True)

    def update(self):
        pos, _ = get_closer_entity(self.blackboard, self.entity, tolerance=self.tolerance)
        if pos is None:
            return Status.FAILURE
        if self.comm_variable:
            set_command(self.blackboard, self.comm_variable, pos)
        return Status.SUCCESS


# ------------------------------ HELPER FUNCTIONS -------------------------------------------------


def get_current_pos(client):
    """Returns the current player position.

    Args:
        client (Client): A client that has read access to the observations namespace.

    Returns:
        A Numpy array with the x, y, z coordinates of the player

    """
    x = read_observation(client, Observations.X_POS)
    y = read_observation(client, Observations.Y_POS)
    z = read_observation(client, Observations.Z_POS)
    return np.array([x, z, y])


def get_current_orientation(client):
    yaw = read_observation(client, Observations.YAW)
    pitch = read_observation(client, Observations.PITCH)
    return np.array([yaw, pitch])


def get_relative_point(client, point):
    cur_pos = get_current_pos(client)
    target_relative = point - cur_pos
    return target_relative


def move_to(client, cur_pos, dest_coords, tolerance):
    """Called within a BT loop, this function performs the actions to move to the given block.
    TODO: Jump when needed

    Args:
        client (Client):           Client able to read observations and write actions.
        cur_pos (np.ndarray):      Current position of the player.
        dest_coords (np.ndarray):  Coordinates in world frame of the destination block,
                                   in the Minecraft block convention.
        tolerance (np.ndarray):    Tolerance for destination reached check.

    Returns:
        Status.SUCCESS if player reached the destination.
        Status.RUNNING if player is moving towards the destination, and the function must be called
                       again to reach it.
        Status.FAILURE if destination is unreachable.

    """
    if within_tolerance_arrays(a=cur_pos, b=dest_coords, tolerance=tolerance):
        return Status.SUCCESS
    look_target = np.array(dest_coords)
    if dest_coords[2] <= cur_pos[2]:
        look_target[2] = np.max([dest_coords[2] + 1.625, cur_pos[2]])
    else:
        look_target[2] = np.min([dest_coords[2] + 1.625, cur_pos[2] + 1.625])

    if look_at(client, look_target):
        set_action(client, Actions.MOVE, 0.5)
        if dest_coords[2] > cur_pos[2]:
            set_action(client, Actions.MOVE, 1)
            set_action(client, Actions.JUMP, 1)
    return Status.RUNNING


def destroy_block(client, block_coords):
    """Called in a BT loop, this function performs the actions to destroy the given block.

    Args:
        client (Client):            Client able to read observations and write actions.
        block_coords (np.ndarray):  Coordinates in world frame of the block to destroy,
                                    in the Minecraft convention block reference.

    Returns:
        Status.SUCCESS if block was destroyed.
        Status.RUNNING if block is being destroyed, and the function must be called again to
                       finish the destruction.
        Status.FAILURE if the block is not within range

    """
    block_grid = get_grid_block(client, block_coords)
    grid = read_observation(client, Observations.GRID)
    try:
        line_sight = read_observation(client, Observations.LINE_OF_SIGHT)
    except KeyError:
        line_sight = None
    target_block_type = grid[block_grid]
    if target_block_type == Items.AIR:
        set_action(client, Actions.ATTACK, 0)
        return Status.SUCCESS  # Block was destroyed.

    tool = get_tool_to_destroy(block=target_block_type)
    if len(tool) == 0:
        set_action(client, Actions.SELECT, Actions.HAND)
    else:
        inventory = read_observation(client, Observations.INVENTORY)
        owned_tool = MalmoEnv.get_item_from_inventory(inventory=inventory, requested_item=tool)
        if owned_tool:
            set_action(client, Actions.SELECT, owned_tool[0]['type'])
        else:
            print('To destroy ' + str(target_block_type) + ' ' + str(tool) + ' is needed!')
            return Status.FAILURE

    if look_at(client, block_center(block_coords)):
        if line_sight is not None and not line_sight['inRange']:
            set_action(client, Actions.ATTACK, 0)
            return Status.FAILURE  # The block we want to hit is not in range
        set_action(client, Actions.ATTACK, 1)
    return Status.RUNNING


def look_at(client, target, tolerance=5, speed_modifier=0.5):
    """Perform the TURN and PITCH actions in order to orient the player towards the target.

    The correct turning speed is computed by taking into consideration the maximum turning speed,
    the average time between decisions, and the game frequency, in order to avoid oscillations.

    Args:
        client (Client): Client of the behavior calling the action
        target (np.ndarray): Position of the target, in the format [x, z, y]
        tolerance (float): Tolerance (in degrees) for stopping the turning action.
        speed_modifier (float): Modifier of the turning speed. A value of 1 means that the player
            will try to turn as fast as possible towards the target, but delays in the
            asynchronous malmo environment may make it oscillate. Values lower than 1 will make
            the turning slower, but less oscillating and can therefore achieve the goal faster.

    Returns:
        True if the player is oriented towards the target within the given tolerance,
        False otherwise

    """
    max_turn_speed = read_observation(client, Observations.TURN_SPEED)  # degrees per ms
    avg_step_duration = read_observation(client, Observations.AVG_STEP_DURATION)
    ms_per_tick = read_observation(client, Observations.MS_PER_TICK)
    orientation = get_current_orientation(client)
    cur_pos = get_current_pos(client) + np.array([0, 0, 1.625])  # Aim origin is one block above
    target_rel = target - cur_pos

    # Computing yaw difference
    yaw = orientation[0] if orientation[0] <= 180 else -(360 - orientation[0])
    target_yaw = - np.degrees(np.arctan2(target_rel[0], target_rel[1]))
    delta_yaw = (target_yaw - yaw + 540) % 360 - 180

    # Computing turning speed to perform turn of delta_yaw in one step
    yaw_speed = 1000 * (np.abs(delta_yaw) / avg_step_duration) / ms_per_tick
    yaw_cmd = speed_modifier * np.clip(yaw_speed / max_turn_speed, 0, 1)

    # Computing pitch difference
    distance = np.linalg.norm(target_rel)
    delta_pitch = orientation[1] - np.degrees(np.arcsin((-target_rel[2]) / distance))

    # Computing turning speed to perform delta_pitch in one step
    pitch_speed = 1000 * (np.abs(delta_pitch) / avg_step_duration) / ms_per_tick
    pitch_cmd = speed_modifier * np.clip(pitch_speed / max_turn_speed, 0, 1)

    if delta_yaw > 0:
        set_action(client, Actions.TURN, yaw_cmd)
    else:
        set_action(client, Actions.TURN, -yaw_cmd)

    if delta_pitch > 0:  # Target is above
        set_action(client, Actions.PITCH, -pitch_cmd)  # Negative pitch command means look up
    else:
        set_action(client, Actions.PITCH, pitch_cmd)

    if np.abs(delta_yaw) < tolerance and np.abs(delta_pitch) < tolerance:
        set_action(client, Actions.TURN, 0)
        return True
    return False


def center_angle(angle):
    """Returns the given angle in the [-180, 180] range.

    Args:
        angle (float): Angle in the [0, 360] range.

    Returns:
        The given angle in the [-180, 180] range. Positive angles clockwise.

    """
    return angle if angle <= 180 else -(360 - angle)


def block_center(block_coords):
    return block_coords + np.array([.5, .5, .5])


def get_grid_block(client, target_coords):
    """Compute the indices of the grid observation correspondent to the target coords.

    Args:
        client (Client):            Client able to read Observations.
        target_coords (np.ndarray): Coordinates of the target point, in the world frame.

    Returns:
        The indices of the grid block that contains the given coordinates.

    """
    target_rel = target_coords - np.floor(get_current_pos(client))
    grid_size = read_observation(client, Observations.GRID_DIM)
    center = (np.array([grid_size['x_size'], grid_size['z_size'], grid_size['y_size']]) - 1) / 2
    grid_coords = center + target_rel
    grid_coords = tuple(np.array(grid_coords, dtype=int))
    return grid_coords


def within_tolerance_arrays(a, b, tolerance):
    """Check whether the two arrays are equal within the given tolerance. 

    Args:
        a (np.ndarray): Numpy array
        b (np.ndarray): Numpy array
        tolerance (np.ndarray): Numpy array that represent the tolerance for each dimension of the
            given arrays

    Returns:
        True if the two arrays are equal within the tolerance, False otherwise

    """
    return np.all(np.abs(a - b) - tolerance <= 0)


def within_tolerance_tuples(a, b, tolerance):
    """Check whether the two tuples are equal within the given tolerance.

    Args:
        a (tuple): Tuple of floats or ints
        b (tuple): Tuple of floats or ints
        tolerance (np.ndarray): Numpy array that represent the tolerance for each dimension of the
            given tuples

    Returns:
        True if the two tuples are equal within the tolerance, False otherwise

    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    return np.all(np.abs(a_arr - b_arr) - tolerance < 0)


def has_item(client, item, quantity):
    inventory = read_observation(client, Observations.INVENTORY)
    available_items = MalmoEnv.get_item_from_inventory(inventory, item)
    if not available_items:
        return False
    tot_quantity = sum(item['quantity'] for item in available_items)
    if quantity is None or tot_quantity >= quantity:
        return True
    return False


def get_closer_entity(client, entity_name, tolerance=None):
    cur_pos = get_current_pos(client)
    entities = read_observation(client, Observations.NEARBY_ENTITIES)
    min_dist = np.inf
    min_dist_entity = None
    min_dist_entity_pos = None
    for entity in entities:
        if entity['name'] == entity_name:
            pos = np.array([entity['x'], entity['z'], entity['y']])
            if tolerance is not None and not within_tolerance_arrays(cur_pos, pos, tolerance):
                continue
            dist = np.linalg.norm(pos - cur_pos)
            if dist < min_dist:
                min_dist_entity = entity
                min_dist_entity_pos = pos
    return min_dist_entity_pos, min_dist_entity


def get_grid_center(grid_dim):
    center_x = int((grid_dim['x_size'] - 1) / 2)
    center_z = int((grid_dim['z_size'] - 1) / 2)
    center_y = int((grid_dim['y_size'] - 1) / 2)
    return np.array([center_x, center_z, center_y], dtype=np.int)


def tuple_dist(a, b):
    a_arr = np.array(a)
    b_arr = np.array(b)
    return np.linalg.norm(a_arr - b_arr)


class AStarNode:

    def __init__(self, previous, state, g):
        """Instantiate a AStar search node.

        Args:
            previous (AStarNode):   The previous node in the search.
            state (tuple):          A tuple (player_pos, removal_list), where player_pos is a tuple
                                    (x, z, y) with the coordinates (in the grid observation frame)
                                    of the player position, and removal list is a list of tuples
                                    that correspond to the blocks destroyed in the previous nodes
                                    along the search path.
            g (float):              Cost from the source state to this state.
        """
        self.previous = previous
        self.state = state
        self.g = g


def compute_path_astar(cur_pos, dest, grid, step_cost, destroy_cost, tolerance):
    """Perform A* search to compute the shortest path from the source block to dest block.

    The path is compute to have as ending block any block within the tolerance from the given
    destination, using the given cost functions.

    Args:
        cur_pos (np.ndarray):    Current position of the player in world coordinates. Used to
                                 convert the path into world coordinates.
        dest (tuple):            A (x, z, y) tuple with the destination block.
        grid (np.ndarray):       The grid observation.
        step_cost (function):    A function f(block, neighbor, grid) -> float, that returns the
                                 cost of taking a single step from a block to its neighbor.
        destroy_cost (function): A function f(block) -> float, that returns the cost of
                                 destroying the given block.
        tolerance (np.ndarray):  Tolerance for checking when dest is reached.

    Returns:
        A list of operations in the form (op_name, target_block) where:
            op_name (str):              'MOVE_TO' or 'DESTROY_BLOCK'.
            target_block (np.ndarray):  The target block coordinates in world frame.

    """
    MAX_COMPUTATION_STEPS = 1000

    cur_pos = np.floor(cur_pos)
    grid_center = (np.array(grid.shape) - 1) / 2
    source = tuple(grid_center)
    dest_blocks = get_neighborhood(dest, size=tuple(tolerance), grid=grid)

    node_id = 0  # Used to avoid conflict in heap for nodes with same cost
    min_dist = min_dist_multiple(dest, dest_blocks)  # Get minimal distance to destinations
    start_node = AStarNode(previous=None, state=(source, []), g=0)
    priority_queue = [(min_dist, node_id, start_node)]
    heapq.heapify(priority_queue)
    path_found = False

    visited_states = set()
    steps = -1
    while True:
        steps += 1
        if steps >= MAX_COMPUTATION_STEPS:
            return None

        tot_cost, _, cur_node = heapq.heappop(priority_queue)  # Pop node with minimal g + h
        if cur_node.state[0] in dest_blocks:
            path_found = True
            break
        state_grid = apply_state(grid, cur_node.state)
        new_nodes = []

        # Compute new states resulting from moving to the current position to a neighbor position
        new_positions = get_neighborhood(cur_node.state[0], size=(1, 1, 1), grid=state_grid)
        for new_pos in new_positions:
            # New state has new_pos as position and the removal list of the current node
            state = (new_pos, cur_node.state[1])
            if str(state) in visited_states:
                continue
            else:
                visited_states.add(str(state))
            c = step_cost(cur_node.state[0], new_pos, state_grid)
            if c is None:  # None means that it's not possible to move from current to new pos
                continue
            _, h = min_dist_multiple(new_pos, dest_blocks)
            new_g = cur_node.g + c
            new_node = (new_g + h, node_id, AStarNode(previous=cur_node, state=state, g=new_g))
            new_nodes.append(new_node)
            node_id += 1

        # Compute new states resulting from destroying a block
        new_destroyed = get_neighborhood(cur_node.state[0], size=(1, 1, 1), grid=state_grid,
                                         exclude_blocks=[Items.AIR])
        for destroyed_block in new_destroyed:
            state = (cur_node.state[0], cur_node.state[1] + [destroyed_block])
            if str(state) in visited_states:
                continue
            else:
                visited_states.add(str(state))
            c = destroy_cost(destroyed_block)
            if c is None:  # None means the block cannot be destroyed
                continue
            _, h = min_dist_multiple(cur_node.state[0], dest_blocks)
            new_g = cur_node.g + c
            # New state has current position and the previous removal list plus the block destroyed
            new_node = (new_g + h, node_id, AStarNode(previous=cur_node, state=state, g=new_g))
            new_nodes.append(new_node)
            node_id += 1

        for new_node in new_nodes:
            heapq.heappush(priority_queue, new_node)
    if not path_found:  # TODO: Behavior tree should react and select a new destination
        raise ValueError('Error: cannot find a valid path!')

    # Building the states path from the leaf towards the root
    path = deque()
    while cur_node:
        path.appendleft(cur_node.state)
        cur_node = cur_node.previous

    # Now we convert the path to a list of instructions where each element is a string and a tuple
    # that represents whether we need to move to that block or to destroy that block
    instructions = []
    for i in range(1, len(path)):
        # If position does not change but removal list does -> DESTROY_BLOCK operation
        if path[i][0] == path[i - 1][0] and len(path[i][1]) > len(path[i - 1][1]):
            instruction = ('DESTROY_BLOCK', path[i][1][-1])  # Last element of removal list
        # If position changes and removal list doesn't -> MOVE_TO operation
        elif path[i][0] != path[i - 1][0] and len(path[i][1]) == len(path[i - 1][1]):
            instruction = ('MOVE_TO', path[i][0])
        else:  # If both change, we have a problem
            raise ValueError('Error: a single step of the computed path involves two operations')
        instructions.append(instruction)

    instructions = [(op, cur_pos + np.array(target) - grid_center) for (op, target) in instructions]
    return instructions


def get_neighborhood(block, size, grid, exclude_blocks=None, exclude_self=True):
    """Get the neighborhood of the given block within the given size -or less if outside the
    boundaries of grid.

    Args:
        block (tuple):          Block of which we want to get the neighborhood.
        size (tuple):           Tuple that, for each dimension i, represents how many blocks
                                to take around the given block. I.e. (1, 1, 1) will take a 3x3
                                cube around the block.
        grid (np.ndarray):      3 dimensional grid observation.
        exclude_blocks (list):  List of block types to be excluded.
        exclude_self (bool):    Whether to exclude the given block from the neighbor list

    Returns:
        A list of tuples with the coordinates of blocks in the neighborhood of the given block,
        which is excluded.

    """
    block = np.array(block, dtype=np.int)
    sizes = [range(-s, s + 1) for s in size]
    neighbor_offsets = np.array(list(it.product(*sizes)), dtype=np.int)
    neighbors = block + neighbor_offsets
    mask_negative = (neighbors > 0).all(axis=1)
    mask_above_limit = (neighbors < grid.shape).all(axis=1)
    mask = mask_negative & mask_above_limit
    if exclude_self:
        mask = mask & (neighbors != 0).all(axis=1)
    if exclude_blocks:
        mask = mask & ~np.isin(grid[tuple(neighbors.T)], exclude_blocks)
    neighbors = neighbors[mask]
    return list(map(tuple, neighbors))


def min_dist_multiple(block, destinations):
    """Compute the distance to the closest -in terms of l2 norm- of the given destinations.

    Args:
        block (tuple):        The source block.
        destinations (list):  List of tuples representing destination blocks

    Returns:
        The minimal distance from source to one of the destinations.

    """
    block = np.array(block, dtype=np.float)
    destinations = np.array(destinations)
    distances = np.sqrt(np.sum((block - destinations)**2, axis=1))
    argmin = np.argmin(distances)
    dist = distances[argmin]
    return argmin, dist


def apply_state(grid, state):
    """Apply the compact representation of a state to the grid.

    Args:
        grid (np.ndarray): The 3 dimensional observation grid.
        state (tuple): A compact representation of a state, (pos, removal_list).

    Returns:
        The given grid to which the given state has been applied.

    """
    state_grid = np.array(grid)
    for removal in state[1]:
        state_grid[removal] = Items.AIR
    return state_grid


def cost_for_move(block, neighbor, grid):
    """Define the cost for moving from block to neighbor in the given grid.

    Args:
        block (tuple):      Starting block.
        neighbor (tuple):   Destination block, must be a 1-neighbor.
        grid (np.ndarray):  The 3 dimensional grid observation.

    Returns:
        The cost for moving from block to neighbor, or None if the movement is not possible.

    """
    # We cannot step into a block if it has air below
    if neighbor[2] == 0 or grid[neighbor[0], neighbor[1], neighbor[2] - 1] == Items.AIR:
        return None
    # We cannot step into a filled block
    if grid[neighbor] != Items.AIR:
        return None
    # We need a block of air above or the player won't fit
    if (neighbor[2] == grid.shape[2] - 1 or
            grid[neighbor[0], neighbor[1], neighbor[2] + 1] != Items.AIR):
        return None
    if neighbor[2] <= block[2]:  # No need to jump
        return 1
    else:  # Need to jump
        return 2


def cost_for_destroy(block):
    """Define the cost for destroying a block.

    Args:
        block (str): The type of the block to be destroyed.

    Returns:
        The cost for destroying the given block, or None if it cannot be destroyed.

    """
    if block == Items.AIR:
        return None
    return 3
