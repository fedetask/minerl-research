import py_trees
from malmo import MalmoPython
import numpy as np
import re
import xml.etree.ElementTree as ET
import json

from Keys import Observations, Actions, Items
from blackboards import register_all_actions, register_all_observations
from blackboards import write_variables, read_variables, Namespace, read_var, set_var


class MalmoEnv:

    def __init__(self, mission_path):
        self.blackboard = py_trees.Client(name='malmo_env')
        register_all_actions(self.blackboard, True)
        register_all_observations(self.blackboard, True)
        self._init_actions()
        self.mission_path = mission_path
        self.agent_host = None
        self.my_mission = None
        self.my_mission_record = None

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

    def run(self, behavior=None, max_steps=0, min_step_duration=15, log_obs=False,
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
            log_obs (bool): Whether to log the observations in the blackboard.
            log_behavior (bool): Whether to print the behavior tree
        """
        step = 0
        done = False
        last_world_time = None
        while not done and (step < max_steps or max_steps <= 0):
            obs = self._read_observations()
            if obs == 'DEAD':
                print('Agent died :(')
                break

            # Checking that we have an observation and that the min_step_duration is respected
            if obs is None:
                continue
            if last_world_time is None:
                last_world_time = obs[Observations.TIME_ALIVE]
            if obs[Observations.TIME_ALIVE] < last_world_time + min_step_duration:
                continue
            last_world_time = obs[Observations.TIME_ALIVE]

            # Main loop observation - action
            self._write_observation(observation=obs)
            if log_obs:
                print(self.blackboard)
            if log_behavior:
                print(py_trees.display.unicode_tree(root=behavior.root, show_status=True))
            if behavior is not None:
                behavior.tick()  # The behavior will write the action in the actions blackboard
            # else: For now we always make a dummy action
            action = self._read_action()
            self._execute_action(action)

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

    def _init_actions(self):
        """Initialize all actions in Keys.Actions with a default value of None. This may be
        needed as the blackboard raises an error when trying to read an uninitialized variable.

        """
        actions = {action: None for action in Actions.all()}
        write_variables(self.blackboard, Namespace.ACTIONS, actions)

    def _write_mission_properties(self):
        """Write in the observations blackboard some properties of the xml that must be taken
        into consideration, such as grid observations dimensions.

        """
        xml_root = ET.parse(self.mission_path).getroot()
        grid = None
        for e in xml_root.iter():
            if bool(re.match(r"{.*}Grid", e.tag)):
                grid = e
        if grid is None:
            raise ValueError('XML Mission does not contain Grid definition')
        min_obj = list(grid)[0]
        max_obj = list(grid)[1]
        grid_dim = {'x_min': int(min_obj.get('x')), 'x_max': int(max_obj.get('x')),
                    'z_min': int(min_obj.get('z')), 'z_max': int(max_obj.get('z')),
                    'y_min': int(min_obj.get('y')), 'y_max': int(max_obj.get('y'))}
        set_var(self.blackboard, Namespace.OBSERVATIONS, Observations.GRID_DIM, grid_dim)

    def _read_observations(self):
        world_state = self.agent_host.getWorldState()
        if world_state.has_mission_begun and not world_state.is_mission_running:
            return 'DEAD'
        if not world_state.has_mission_begun or len(world_state.observations) == 0:
            return None
        observations = json.loads(world_state.observations[-1].text)
        if Observations.GRID in observations:
            observations = self._fix_observation(observations)
        return observations

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
        grid_dim = read_var(self.blackboard, Namespace.OBSERVATIONS, Observations.GRID_DIM)
        print(grid_dim)
        x_size = grid_dim['x_max'] - grid_dim['x_min'] + 1
        z_size = grid_dim['z_max'] - grid_dim['z_min'] + 1
        y_size = grid_dim['y_max'] - grid_dim['y_min'] + 1
        grid = grid_list.reshape((x_size, z_size, y_size), order='F')
        observation[Observations.GRID] = grid
        return observation

    def _execute_action(self, action):
        # Some actions need to be stopped, otherwise they will continue to be executed
        reset_action = {
            'move': 0,
            'strafe': 0,
            'pitch': 0,
            'turn': 0,
            'jump': 0,
            # 'crouch': 0, Resetting crouch can lead to falling
            # 'attack': 0, Resetting attack can prevent destroying blocks
            'use': 0
        }
        for key, value in reset_action.items():
            cmd = self._generate_command(action_name=key, action_value=value)
            self.agent_host.sendCommand(cmd)

        # Executing the actual action
        for key, value in action.items():
            cmd = self._generate_command(action_name=key, action_value=value)
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
            return ''
        if action_name == Actions.SELECT:
            index = self._get_index_in_inventory(action_value)
            return 'swapInventoryItems 0 ' + str(index)
        if action_name == Actions.CRAFT:
            craft_target = self._get_craft_target(action_value)
            if craft_target is not None:
                return Actions.CRAFT + ' ' + craft_target
            else:
                print('Error! Cannot craft ' + action_value)
                return ''
        return action_name + ' ' + str(action_value)

    def _get_index_in_inventory(self, requested_item):
        for item in self.blackboard.observations.inventory:
            if item['type'] == requested_item:
                return item['index']
        return None

    def _get_craft_target(self, item):
        """Returns the correct name for the Malmo command to craft the desired item.

        This comes necessary when crafting items that have multiple variants of which we are not
        interested. For example, get_craft_target(Items.PLANKS) will return the type of planks
        corresponding to the type of wood in the inventory.

        Returns:
            A string with the correct name for the item to craft

        """
        if item == Items.PLANKS:
            for it in self.blackboard.observations.inventory:
                if it['type'] == Items.LOG:
                    return it['variant'] + ' ' + Items.PLANKS
            return None
        else:
            return item


if __name__ == '__main__':
    import subtrees

    env = MalmoEnv()
    env.init()
    behavior_tree = subtrees.get_behavior_tree()
    env.run(behavior=behavior_tree, max_steps=0, min_step_duration=20, log_obs=False,
            log_behavior=True)
