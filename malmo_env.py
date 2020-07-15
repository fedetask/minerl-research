from abc import ABC, abstractmethod
from blackboards import register_variables, write_variables, read_variables, Namespace
from py_trees.behaviour import Behaviour
from py_trees.blackboard import Client
from Keys import Observations, Actions, Items


class MalmoEnv(ABC):

    def __init__(self):
        self.blackboard = Client(name='malmo_env')
        register_variables(self.blackboard, namespace=Namespace.OBSERVATIONS,
                           can_write=True, variables=Observations.all())
        register_variables(self.blackboard, namespace=Namespace.ACTIONS,
                           can_write=True, variables=Actions.all())

    def run(self, behavior=None, max_steps=0, min_step_duration=15, log_obs=False):
        """Run the main loop of the environment.

        The loop consists of the following steps:
            1. Acquire the observation and writes the variables defined in MalmoEnv.observation_keys
            into the observations blackboard.
            2. Tick the behavior.
            3. Execute the action defined in the actions blackboard.

        Args:
            behavior (Behaviour): The behavior that interacts with the Malmo environment.
            max_steps (int): The maximum number of steps (in world time) to run.
            min_step_duration (int): Minimum number of steps (in world time) between subsequent
                observations. Due to the asynchronous nature a fixed number of steps cannot be
                enforced.
            log_obs (bool): Whether to log the observations in the blackboard
        """
        step = 0
        done = False
        last_world_time = None
        while not done and (step < max_steps or max_steps <= 0):
            obs = self.read_observations()
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
            self.write_observation(observation=obs)
            if log_obs:
                print(self.blackboard)
            if behavior is not None:
                behavior.tick()  # The behavior will write the action in the actions blackboard
            else:
                self.dummy_action()  # Writing a dummy action in the blackboard (testing purpose)
            action = self.read_action()
            self.execute_action(action)

    def write_observation(self, observation):
        """Extract from the observations dictionary the key-value pairs defined in
        MalmoEnv.observation_keys and write them in the observations blackboard.
        Args:
            observation (dict): A dictionary of the observed variables and their respective values.

        """
        to_write = {k: observation[k] for k in Observations.all()}
        write_variables(self.blackboard, Namespace.OBSERVATIONS, to_write)

    def read_action(self):
        """Read the action from the actions blackboard.

        Returns:
            A dictionary containing the action to perform.

        """
        return read_variables(self.blackboard, Namespace.ACTIONS, Actions.all())

    cont = 0
    def dummy_action(self):
        variables = {
            'move': 1 if MalmoEnv.cont % 2 == 0 else 0,
            'strafe': 0,
            'pitch': 0,
            'turn': 1 if MalmoEnv.cont % 2 == 0 else 0,
            'jump': 0,
            'crouch': 0,
            'attack': 1,
            'use': 0,
            'select': Items.DIRT,
            'craft': Items.PLANKS if MalmoEnv.cont % 2 == 0 else Items.STICK
        }
        MalmoEnv.cont += 1
        write_variables(self.blackboard, Namespace.ACTIONS, variables)

    @abstractmethod
    def init(self):
        """Perform all the operations necessary to make the environment ready to be used and
        start the mission.

        """
        pass

    @abstractmethod
    def read_observations(self):
        """Read the world state and return it as a dictionary with keys defined in
        MalmoEnv.observation_keys.

        Returns:
            A dictionary with the observed variables

        """

    @abstractmethod
    def execute_action(self, action):
        """Execute the given action in the Malmo environment.

        Args:
            action (dict): A dictionary that encodes the action to execute. Action keys are
            defined in MalmoEnv.action_keys.

        """
