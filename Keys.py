from abc import ABC, abstractmethod


class KeyList(ABC):
    """Defines a class that will only contain variables corresponding to Malmo keys.

    """

    @staticmethod
    @abstractmethod
    def all():
        """
        Returns: A list containing all the variable values of the KeyList

        """
        pass


class Observations(KeyList):
    TIME_ALIVE = 'TimeAlive'
    LIFE = 'Life'
    SCORE = 'Score'
    FOOD = 'Food'
    IS_ALIVE = 'IsAlive'
    AIR = 'Air'
    X_POS = 'XPos'
    Y_POS = 'YPos'
    Z_POS = 'ZPos'
    PITCH = 'Pitch'
    YAW = 'Yaw'
    NEARBY_ENTITIES = 'NearbyEntities'
    INVENTORY = 'inventory'
    GRID = 'Grid'
    GRID_DIM = 'Grid_dim'
    AVG_STEP_DURATION = 'AvgStepDuration'
    TURN_SPEED = 'TurnSpeed'
    MS_PER_TICK = 'MsPerTick'
    LINE_OF_SIGHT = 'LineOfSight'

    @staticmethod
    def all():
        return [Observations.TIME_ALIVE, Observations.LIFE, Observations.SCORE, Observations.FOOD,
                Observations.IS_ALIVE, Observations.AIR, Observations.X_POS, Observations.Y_POS,
                Observations.Z_POS, Observations.PITCH, Observations.YAW,
                Observations.NEARBY_ENTITIES, Observations.INVENTORY, Observations.GRID,
                Observations.GRID_DIM, Observations.AVG_STEP_DURATION, Observations.TURN_SPEED,
                Observations.MS_PER_TICK, Observations.LINE_OF_SIGHT]


class Items(KeyList):
    DIRT = 'dirt'
    STICK = 'stick'
    PLANKS = 'planks'
    CRAFTING_TABLE = 'crafting_table'
    WOODEN_PICKAXE = 'wooden_pickaxe'
    STONE_PICKAXE = 'stone_pickaxe'
    COBBLESTONE = 'cobblestone'
    FURNACE = 'furnace'
    LOG = 'log'
    AIR = 'air'

    @staticmethod
    def all():
        return [Items.DIRT, Items.STICK, Items.PLANKS, Items.CRAFTING_TABLE,
                Items.WOODEN_PICKAXE, Items.STONE_PICKAXE, Items.COBBLESTONE, Items.FURNACE,
                Items.LOG, Items.AIR]


class Actions(KeyList):
    MOVE = 'move'
    STRAFE = 'strafe'
    PITCH = 'pitch'
    TURN = 'turn'
    JUMP = 'jump'
    CROUCH = 'crouch'
    ATTACK = 'attack'
    USE = 'use'
    SELECT = 'select'
    CRAFT = 'craft'

    @staticmethod
    def all():
        return [Actions.MOVE, Actions.STRAFE, Actions.PITCH, Actions.TURN, Actions.JUMP,
                Actions.CROUCH, Actions.ATTACK, Actions.USE, Actions.SELECT, Actions.CRAFT]


class Commands(KeyList):

    CLOSEST_WOOD = 'closest_wood'

    @staticmethod
    def all():
        return [Commands.CLOSEST_WOOD]
