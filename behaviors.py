from py_trees.behaviour import Behaviour
from py_trees.blackboard import Client
from py_trees.common import Status
from blackboards import Namespace, read_command, set_command, set_action, read_observation
from blackboards import register_actions, register_commands, register_observations

import numpy as np
import time

from Keys import Observations, Actions, Commands, Items
from malmo_env import MalmoEnv
from crafting import can_craft


# --------------------------------- Behaviors ----------------------------------------------

class MoveTo(Behaviour):
    """Behavior that moves the player towards the destination specified in the commands blackboard

    """

    def __init__(self, name, destination_key, tolerance):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.destination_key = destination_key
        self.tolerance = tolerance

    def setup(self):
        super().setup()
        register_actions(client=self.blackboard, can_write=True)
        register_commands(client=self.blackboard, can_write=False)
        register_observations(client=self.blackboard, can_write=False)

    def terminate(self, new_status):
        super().terminate(new_status)
        set_action(self.blackboard, Actions.MOVE, 0)

    def update(self):
        destination = read_command(self.blackboard, self.destination_key)
        cur_pos = get_current_pos(self.blackboard)
        if is_at(cur_pos=cur_pos, destination=destination, tolerance=self.tolerance):
            return Status.SUCCESS
        if look_at(self.blackboard, destination, tolerance=5):
            set_action(self.blackboard, Actions.MOVE, 0.5)
        return Status.RUNNING


class IsAt(Behaviour):
    """Behavior that checks whether the player is at the destination specified in the commands
    blackboard with the desired level of tolerance.

    """

    def __init__(self, name, destination_key, tolerance=1.):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.destination_key = destination_key
        self.destination = None
        self.tolerance = tolerance

    def setup(self):
        super().setup()
        register_actions(client=self.blackboard, can_write=True)
        register_commands(client=self.blackboard, can_write=False)
        register_observations(client=self.blackboard, can_write=False)

    def initialise(self):
        super().initialise()
        self.destination = read_command(self.blackboard, self.destination_key)

    def update(self):
        cur_pos = get_current_pos(self.blackboard)
        if is_at(cur_pos=cur_pos, destination=self.destination, tolerance=self.tolerance):
            return Status.SUCCESS
        else:
            print('Not reached: ' + str(cur_pos))
            return Status.FAILURE


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
        center_x = int((grid_dim['x_size'] - 1) / 2)
        center_z = int((grid_dim['z_size'] - 1) / 2)
        center_y = int((grid_dim['y_size'] - 1) / 2)
        if self.tol is None or self.tol[0] is None:
            self.x_lim = [0, grid_dim['x_size']]
        else:
            self.x_lim = [center_x - self.tol[0], center_x + self.tol[0] + 1]
        if self.tol is None or self.tol[1] is None:
            self.z_lim = [0, grid_dim['z_size']]
        else:
            self.z_lim = [center_z - self.tol[1], center_z + self.tol[1] + 1]
        if self.tol is None or self.tol[2] is None:
            self.y_lim = [0, grid_dim['y_size']]
        else:
            self.y_lim = [center_y - self.tol[2], center_y + self.tol[2] + 1]

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
        self.target_block = None

    def setup(self, **kwargs):
        super().setup(**kwargs)
        register_observations(self.blackboard, can_write=False)
        register_actions(self.blackboard, can_write=True)
        register_commands(self.blackboard, can_write=False)

    def initialise(self):
        super().initialise()
        self.target_coords = read_command(self.blackboard, self.comm_variable)
        self.target_block = get_grid_block(self.blackboard, self.target_coords)
        if read_observation(self.blackboard, Observations.GRID)[self.target_block] == Items.AIR:
            print('ERROR: TARGETING AIR BLOCK')

    def terminate(self, new_status):
        super().terminate(new_status)
        self.target_coords = None
        self.target_block = None
        set_action(self.blackboard, Actions.ATTACK, 0)

    def update(self):
        grid = read_observation(self.blackboard, Observations.GRID)
        line_sight = read_observation(self.blackboard, Observations.LINE_OF_SIGHT)

        if grid[self.target_block] == Items.AIR:
            set_action(self.blackboard, Actions.ATTACK, 0)
            return Status.SUCCESS  # Block was destroyed. TODO: Check it was not air already

        if look_at(self.blackboard, block_center(self.target_coords), tolerance=3):
            if not line_sight['inRange']:
                set_action(self.blackboard, Actions.ATTACK, 0)
                return Status.FAILURE  # The block we want to hit is not in range
            set_action(self.blackboard, Actions.ATTACK, 1)
        return Status.RUNNING


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
        pos, _ = get_closer_entity(self.blackboard, self.entity)
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


def look_at(client, target, tolerance, speed_modifier=0.5):
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


def block_center(block_coords):
    return block_coords + np.array([.5, .5, .5])


def get_grid_block(client, target_coords):
    """Compute the indices of the grid observation correspondent to the target coords.

    Args:
        client (Client): Client able to read Observations.
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


def is_at(cur_pos, destination, tolerance):
    return np.all(np.abs(cur_pos - destination) - tolerance < 0)


def has_item(client, item, quantity):
    inventory = read_observation(client, Observations.INVENTORY)
    available_items = MalmoEnv.get_item_from_inventory(inventory, item)
    if not available_items:
        return False
    tot_quantity = sum(item['quantity'] for item in available_items)
    if quantity is None or tot_quantity >= quantity:
        return True
    return False


def get_closer_entity(client, entity_name):
    cur_pos = get_current_pos(client)
    entities = read_observation(client, Observations.NEARBY_ENTITIES)
    min_dist = np.inf
    min_dist_entity = None
    min_dist_entity_pos = None
    for entity in entities:
        if entity['name'] == entity_name:
            pos = np.array([entity['x'], entity['z'], entity['y']])
            dist = np.linalg.norm(pos - cur_pos)
            if dist < min_dist:
                min_dist_entity = entity
    return min_dist_entity_pos, min_dist_entity