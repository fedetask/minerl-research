import py_trees
from malmo import MalmoPython
import numpy as np
import re
import xml.etree.ElementTree as ET
import json
from collections import deque

from Keys import Observations, Actions, Items
from blackboards import register_actions, register_observations
from blackboards import write_variables, read_variables, Namespace
from blackboards import read_observation, set_observation, read_action, set_action


class MalmoEnv:

    def __init__(self, mission_path):
        self.blackboard = py_trees.blackboard.Client(name='malmo_env')
        register_actions(self.blackboard, True)
        register_observations(self.blackboard, True)
        self.mission_path = mission_path
        self.agent_host = None
        self.my_mission = None
        self.my_mission_record = None
        self.init_steps = 2
        self.step_durations = deque(maxlen=10)

    def init(self):
        """Read the mission XML file and instantiate the malmo communication variables.

        """
        with open(self.mission_path) as f:
            mission_content = f.read()
        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(mission_content, True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()
        self.agent_host.startMission(self.my_mission, self.my_mission_record)
        print('Mission started')
        self._write_mission_properties()
        self._reset_actions()

    def run(self, behavior=None, max_steps=0, min_step_duration=15, log_obs=[],
            log_behavior=False):
        """Run the main loop of the environment.

        The loop consists of the following steps:
            1. Acquire the observation and writes the variables defined in MalmoEnv.observation_keys
            into the observations blackboard.
            2. Tick the behavior.
            3. Execute the action defined in the actions blackboard.

        Args:
            behavior (py_trees.trees.BehaviourTree): The behavior that interacts with the Malmo
            environment.
            max_steps (int): The maximum number of steps (in world time) to run.
            min_step_duration (int): Minimum number of steps (in world time) between subsequent
                observations. Due to the asynchronous nature a fixed number of steps cannot be
                enforced.
            log_obs (list): List of observation keys (from Keys.Observations) to log at each step.
            log_behavior (bool): Whether to print the behavior tree
        """
        step = 0
        done = False
        last_world_time = None
        while not done and (step < max_steps or max_steps <= 0):
            obs = self._read_observation()
            if 'MISSION_ENDED' in obs:
                print('Agent died :(')
                break

            # Checking that we have an observation and that the min_step_duration is respected
            if 'EMPTY_OBSERVATION' in obs:
                continue
            if last_world_time is None:
                last_world_time = obs[Observations.TIME_ALIVE]
            step_duration = obs[Observations.TIME_ALIVE] - last_world_time
            if step_duration < min_step_duration:
                continue
            last_world_time = obs[Observations.TIME_ALIVE]

            # Here we are in a valid step
            step += 1
            self.step_durations.append(step_duration)
            if step_duration > min_step_duration * 1.1:
                print('Cannot keep up with environment: step_duration was ' + str(step_duration)
                      + ' while target is ' + str(min_step_duration))
            obs = self._fix_observation(observation=obs)
            obs[Observations.AVG_STEP_DURATION] = np.mean(self.step_durations)
            self._write_observation(observation=obs)

            if step <= self.init_steps:  # In the init steps we jump because it frequently
                # happens that the player spawns inside the terrain.
                if step < self.init_steps:
                    init_action = {Actions.JUMP: 1}
                else:
                    init_action = {Actions.JUMP: 0}
                self._execute_action(init_action)
                self._reset_actions()
                continue

            if behavior is not None:
                behavior.tick()  # The behavior will write the action in the actions blackboard
            action = self._read_action()
            self._execute_action(action)
            self._log(log_obs, log_behavior, obs, behavior)
            self._reset_actions(exclude_mask=[Actions.CROUCH, Actions.ATTACK])

    @staticmethod
    def get_item_from_inventory(inventory, requested_item, variant=None):
        """Get the requested item from the inventory.

        Args:
            inventory (dict): The inventory dictionary
            requested_item (str): Key corresponding to the item
            variant (str): The requested variant. If None, all owned variants will be returned.

        Returns:
            The first instance of the requested item, or None if not found

        """
        available_items = []
        for item in inventory:
            if item['type'] == requested_item and (variant is None or item['variant'] == variant):
                available_items.append(item)
        return available_items

    def _write_mission_properties(self):
        """Write in the observations blackboard some properties of the xml that must be taken
        into consideration.

        Properties written:
            - Observations.GRID_DIM: A dictionary encoding the size of the maximum observable grid
            - Observations.TURN_SPEED: The speed of turning with the PITCH and YAW actions

        """
        # Writing grid properties
        xml_root = ET.parse(self.mission_path).getroot()
        grid = None
        for e in xml_root.iter():
            if bool(re.match(r"{.*}Grid", e.tag)):
                grid = e
                break
        assert grid is not None, 'XML Mission does not contain Grid definition'
        min_obj = list(grid)[0]
        max_obj = list(grid)[1]
        x_left, x_right = int(min_obj.get('x')), int(max_obj.get('x'))
        z_left, z_right = int(min_obj.get('z')), int(max_obj.get('z'))
        y_left, y_right = int(min_obj.get('y')), int(max_obj.get('y'))
        assert x_left == -x_right and z_left == -z_right and y_left == -y_right, \
            "Error: grid min and max must be symmetrical"
        grid_dim = {'x_size': x_right - x_left + 1,
                    'z_size': z_right - z_left + 1,
                    'y_size': y_right - y_left + 1
                    }
        set_observation(self.blackboard, Observations.GRID_DIM, grid_dim)

        # Writing continuous commands properties
        turn_speed = None
        for e in xml_root.iter():
            if bool(re.match(r"{.*}ContinuousMovementCommands", e.tag)):
                turn_speed = int(e.get('turnSpeedDegs'))
                break
        assert turn_speed is not None, 'XML Mission does not contain continuous commands turn speed'
        set_observation(self.blackboard, Observations.TURN_SPEED, turn_speed)

        # Write mod settings
        ms_per_tick = None
        for e in xml_root.iter():
            if bool(re.match(r"{.*}MsPerTick", e.tag)):
                ms_per_tick = int(e.text)
        assert ms_per_tick is not None, 'XML Mission does not contain MsPerTick'
        set_observation(self.blackboard, Observations.MS_PER_TICK, ms_per_tick)

    def _reset_actions(self, exclude_mask=[]):
        """Reset the actions blackboard by setting all actions to None, excluding those in the
        exclude_mask.

        Args:
            exclude_mask (list): List of action names that will not be reset to None.

        """
        actions = {action: None for action in Actions.all() if action not in exclude_mask}
        write_variables(self.blackboard, Namespace.ACTIONS, actions)

    def _read_observation(self):
        world_state = self.agent_host.getWorldState()
        if world_state.has_mission_begun and not world_state.is_mission_running:
            return {'MISSION_ENDED'}
        if not world_state.has_mission_begun or len(world_state.observations) == 0:
            return {'EMPTY_OBSERVATION'}
        observation = json.loads(world_state.observations[-1].text)
        if Observations.TIME_ALIVE not in observation:
            return {'EMPTY_OBSERVATION'}
        return observation

    def _fix_observation(self, observation):
        """Transform the observation into manageable types. Currently transforms:
        'Grid': from list to numpy array where the first dimension is the x axis, the second is
        the y axis, and the third is the z axis

        Args:
            observation (dict): The observation read from malmo

        Returns:
            A dictionary with the transformed observations

        """
        grid_list = np.array(observation[Observations.GRID])
        grid_dim = read_observation(self.blackboard, Observations.GRID_DIM)
        grid = grid_list.reshape((grid_dim['x_size'], grid_dim['z_size'], grid_dim['y_size']),
                                 order='F')
        observation[Observations.GRID] = grid
        return observation

    def _write_observation(self, observation):
        """Extract from the observations dictionary the key-value pairs defined in
        MalmoEnv.observation_keys and write them in the observations blackboard.
        Args:
            observation (dict): A dictionary of the observed variables and their respective values.

        """
        to_write = {k: observation[k] for k in observation if k in Observations.all()}
        write_variables(self.blackboard, Namespace.OBSERVATIONS, to_write)

    def _read_action(self):
        """Read the action from the actions blackboard.

        Returns:
            A dictionary containing the action to perform.

        """
        return read_variables(self.blackboard, Namespace.ACTIONS, Actions.all())

    def _execute_action(self, action):
        """Execute the given action by sending the appropriate commands to the Malmo server.

        Args:
            action (dict): Dictionary of action key-values. Actions with value None will be
            silently discarded.

        """
        for key, value in action.items():
            if value is not None:
                cmd = self._generate_command(action_name=key, action_value=value)
                if cmd:
                    self.agent_host.sendCommand(cmd)

    def _generate_command(self, action_name, action_value):
        """Generate the command to send to the Malmo platform from the given action name and
        action value.

        Args:
            action_name (str): The name of the action, must belong to MalmoEnv.action_keys.
            action_value (Union[str, int]): The value of the action.

        Returns:
            The string command for the Malmo platform to execute the desired action.

        """
        if action_value is None:
            raise ValueError('Error: Crafting command with empty target')
        if action_name == Actions.SELECT:
            available_items = MalmoEnv.get_item_from_inventory(
                inventory=read_observation(self.blackboard, Observations.INVENTORY),
                requested_item=action_value
            )
            if not available_items:
                raise ValueError('Error: Trying to select ' + action_value +
                                 'but it is not present in inventory')
            index = available_items[0]['index']  # The first available is okay
            return 'swapInventoryItems 0 ' + str(index)
        if action_name == Actions.CRAFT:
            if action_value == Items.PLANKS:
                crafting_target = self._get_craft_target(Items.PLANKS)
                if crafting_target is not None:
                    return Actions.CRAFT + ' ' + crafting_target
                else:
                    return ''
        return action_name + ' ' + str(action_value)

    def _get_craft_target(self, item):
        """Returns the extended name of the desired item from the available crafting materials.

            This comes necessary when crafting items that have multiple variants of which we are not
            interested. For example, get_craft_target(Items.PLANKS) will return the type of planks
            corresponding to the type of wood in the inventory.

            Args:
                item (str): The name of the item to craft, as defined in Keys.Items

            Returns:
                A string with the correct name for the item to craft.

            """
        if item == Items.PLANKS:
            inventory = read_observation(self.blackboard, Observations.INVENTORY)
            available_items = MalmoEnv.get_item_from_inventory(inventory, Items.LOG)
            if available_items:
                log = available_items[0]  # The first available is okay
                return log['variant'] + ' ' + Items.PLANKS
            else:
                return None
        else:
            return item

    @staticmethod
    def _log(log_obs, log_behavior, obs, behavior):
        """Log observations and the behavior tree accordingly to the given flags

        Args:
            log_obs (list): List of observation keys to log, or None.
            log_behavior (bool): Whether to print the behavior tree.
            obs (dict): The observations dictionary.
            behavior (py_trees.trees.BehaviourTree): The behavior tree to print.

        """
        if log_obs:
            for obs_key in log_obs:
                print(obs_key + ': ' + str(obs[obs_key]))
        if log_behavior:
            print(py_trees.display.unicode_tree(root=behavior.root, show_status=True))


if __name__ == '__main__':
    import subtrees

    env = MalmoEnv(mission_path='missions/default_world_1.xml')
    env.init()
    behavior_tree = subtrees.get_behavior_tree()
    env.run(behavior=behavior_tree, max_steps=0, min_step_duration=10,
            log_obs=[], log_behavior=False)
