from py_trees.composites import Selector, Sequence
from py_trees.trees import BehaviourTree
from py_trees.blackboard import Client
from py_trees.idioms import eternal_guard
import numpy as np

import behaviors
from Keys import Commands, Actions, Observations, Items
from blackboards import register_commands, write_variables, Namespace


def get_behavior_tree():
    root = get_has_get_wood_pickaxe(name='HasGetWoodPickaxe')
    tree = BehaviourTree(root=root)
    tree.setup()
    blackboard = Client(name='BehaviorTree')
    register_commands(blackboard, can_write=True)
    init_commands = {k: None for k in Commands.all()}
    write_variables(blackboard, Namespace.COMMANDS, init_commands)
    return tree


def get_has_get_wood_pickaxe(name):
    root = Selector(name=name + '_root')
    has_pickaxe = behaviors.HasItem(name=name + '_has_wooden_pickaxe', item=Items.WOODEN_PICKAXE)
    sequence = Sequence(name=name + '_craft_pickaxe_sequence')
    get_wood = get_has_get_wood_subtree(name=name + '_get_wood', quantity=3)
    craft_pickaxe = get_craft_wood_pickaxe_subtree(name=name + '_craft_wood_pickaxe')

    sequence.add_children([get_wood, craft_pickaxe])
    root.add_children([has_pickaxe, sequence])
    return root


def get_craft_wood_pickaxe_subtree(name):
    root = Sequence(name=name + '_root')

    # Has/Get sticks subtree
    sticks_root = Selector(name=name + '_sticks_root')
    has_sticks = behaviors.HasItem(name=name + '_has_sticks', item=Items.STICK, quantity=2)
    craft_sticks_sequence = Sequence(name=name + '_craft_sticks_root')
    has_get_planks_for_sticks = get_has_craft_item_subtree(name=name + '_has_get_planks_for_stick',
                                                           item=Items.PLANKS, quantity=2)
    craft_sticks = behaviors.Craft(name=name + '_craft_sticks', item=Items.STICK)
    sticks_root.add_children([has_sticks, craft_sticks_sequence])
    craft_sticks_sequence.add_children([has_get_planks_for_sticks, craft_sticks])

    # Has/Get planks subtree
    has_get_planks = get_has_craft_item_subtree(name=name + '_has_get_planks', item=Items.PLANKS,
                                                quantity=3)

    # Craft the wooden pickaxe
    craft_wood_pickaxe = behaviors.Craft(name=name + '_craft_wood_pickaxe',
                                         item=Items.WOODEN_PICKAXE)

    # Putting everything together
    root.add_children([sticks_root, has_get_planks, craft_wood_pickaxe])
    return root


def get_has_craft_item_subtree(name, item, quantity):
    root = Selector(name=name + '_root')
    has_item = behaviors.HasItem(name=name + '_has', item=item, quantity=quantity)
    craft_item = behaviors.Craft(name=name + '_craft', item=item)
    root.add_children([has_item, craft_item])
    return root


def get_has_get_wood_subtree(name, quantity):
    root = Selector(name=name + '_root')
    has_wood = behaviors.HasItem(name=name + '_has_logs', item=Items.LOG, quantity=quantity)
    sequence = Sequence(name=name + '_chop_sequence')
    is_get_close_to_wood = get_is_get_close_to_subtree(name=name + '_IsGetCloseTo_wood',
                                                       item=Items.LOG,
                                                       tolerance=np.array([2, 2, 2]),
                                                       comm_variable=Commands.CLOSEST_WOOD)
    chop_wood = behaviors.Destroy(name=name + 'ChopWood', comm_variable=Commands.CLOSEST_WOOD)
    sequence.add_children([is_get_close_to_wood, chop_wood])
    root.add_children([has_wood, sequence])
    return root


def get_is_get_close_to_subtree(name, item, tolerance, comm_variable):
    root = Selector(name + '_root')
    is_close = behaviors.IsCloseTo(name + '_precondition', item=item, tolerance=tolerance,
                                   out_variable=comm_variable)
    has_nearby = behaviors.IsCloseTo(name + '_has_nearby', item=item, tolerance=None,
                                     out_variable=comm_variable)
    move_to = get_move_to_subtree(name + '_MoveTo', destination_key=comm_variable,
                                  tolerance=tolerance)
    move_to_guard = eternal_guard(subtree=move_to, name=name + '_MoveTo_guard',
                                  conditions=[has_nearby])
    root.add_children([is_close, move_to_guard])
    return root


def get_move_to_subtree(name, destination_key, tolerance):
    """Create a MoveTo subtree that has the goal of moving the player to the destination specified
    by the given destination key.

    Args:
        name (str): Unique name of the subtree.
        destination_key (str): The key that uniquely identifies a variable in the commands
            blackboard, and tells the MoveTo subtree from which variable to read its destination.
            It must therefore belong to Keys.Commands
        tolerance (int): Tolerance for destination check.

    Returns:
        The root of the MoveTo subtree

    """
    root = Selector(name + '_root')
    is_at = behaviors.IsAt(name + '_is_at', destination_key=destination_key,
                           tolerance=tolerance)
    go_to = behaviors.GoTo(name + '_goto', destination_key=destination_key)
    root.add_children([is_at, go_to])
    return root
