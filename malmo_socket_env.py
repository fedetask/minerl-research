from malmo import MalmoPython
from malmo_env import MalmoEnv
import json
from Keys import Actions, Observations, Items


class MalmoSocketEnv(MalmoEnv):

    def __init__(self, mission_path='missions/default_world_1.xml', log_obs=True):
        super(MalmoSocketEnv, self).__init__(log_obs=log_obs)
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

    def read_observations(self):
        world_state = self.agent_host.getWorldState()
        if world_state.has_mission_begun and not world_state.is_mission_running:
            return 'DEAD'
        if not world_state.has_mission_begun or len(world_state.observations) == 0:
            return None
        observations = json.loads(world_state.observations[-1].text)
        return observations

    def execute_action(self, action):
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
            cmd = self.generate_command(action_name=key, action_value=value)
            self.agent_host.sendCommand(cmd)

        # Executing the actual action
        for key, value in action.items():
            cmd = self.generate_command(action_name=key, action_value=value)
            self.agent_host.sendCommand(cmd)

    def generate_command(self, action_name, action_value):
        """Generate the command to send to the Malmo platform from the given action name and
        action value.

        Args:
            action_name (str): The name of the action, must belong to MalmoEnv.action_keys.
            action_value (str): The value of the action.

        Returns:
            The string command for the Malmo platform to execute the desired action.

        """
        if action_name == Actions.SELECT:
            index = self.get_index_in_inventory(action_value)
            return 'swapInventoryItems 0 ' + str(index)
        if action_name == Actions.CRAFT:
            craft_target = self.get_craft_target(action_value)
            if craft_target is not None:
                return Actions.CRAFT + ' ' + craft_target
            else:
                print('Error! Cannot craft ' + action_value)
                return ''
        return action_name + ' ' + str(action_value)

    def get_index_in_inventory(self, requested_item):
        for item in self.blackboard.observations.inventory:
            if item['type'] == requested_item:
                return item['index']
        return None

    def get_craft_target(self, item):
        """Returns the correct name for the Malmo command to craft the desired item.

        This comes necessary when crafting items that have multiple variants of which we are not
        interested. For example, get_craft_target(Items.PLANKS) will return the type of planks
        corresponding to the type of wood in the inventory.

        Returns:
            A string with the correct name for the item to craft

        """
        if item == Items.PLANKS:
            for it in self.blackboard.observations.inventory:
                if it['type'] == 'log':
                    return it['variant'] + ' ' + Items.PLANKS
            return None
        else:
            return item


if __name__ == '__main__':
    env = MalmoSocketEnv(log_obs=False)
    env.init()
    env.run(max_steps=0, min_step_duration=20)
