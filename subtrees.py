"""This module contains all the functions that build the behavior tree using the py_trees library,
    extending it by adding a MemorylessSequence node.

"""

from py_trees.composites import Selector, Sequence, Composite
from py_trees.decorators import FailureIsSuccess
from py_trees.trees import BehaviourTree
from py_trees.blackboard import Client
from py_trees.idioms import eternal_guard
from py_trees.common import Status
import numpy as np
import itertools

import behaviors
from Keys import Commands, Actions, Observations, Items
from blackboards import register_commands, write_variables, Namespace


CLOSE_TO_BLOCK = np.array([2, 2, 4])
CLOSE_TO_FLOATING = np.array([1, 1, 1])


class MemorylessSequence(Composite):
    """Implements a Sequence node that always ticks children from the first to the first RUNNING or
        FAILURE
    """
    def __init__(self, name="Sequence", children=None):
        super(MemorylessSequence, self).__init__(name, children)

    def tick(self):
        """
        Tick over the children.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)

        # reset
        self.logger.debug("%s.tick() [!RUNNING->resetting child index]" % self.__class__.__name__)
        self.current_child = self.children[0] if self.children else None
        for child in self.children:
            # reset the children
            if child.status != Status.INVALID:
                child.stop(Status.INVALID)
        # subclass (user) handling
        self.initialise()

        # customised work
        self.update()

        # nothing to do
        if not self.children:
            self.current_child = None
            self.stop(Status.SUCCESS)
            yield self
            return

        # iterate through children
        index = self.children.index(self.current_child)
        for child in itertools.islice(self.children, index, None):
            for node in child.tick():
                yield node
                if node is child and node.status != Status.SUCCESS:
                    self.status = node.status
                    yield self
                    return
            try:
                # advance if there is 'next' sibling
                self.current_child = self.children[index + 1]
            except IndexError:
                pass

        self.stop(Status.SUCCESS)
        yield self


def get_behavior_tree():
    root = get_has_get_stone_pickaxe(name='HasGetStonePickaxe', quantity=1)
    tree = BehaviourTree(root=root)
    tree.setup()
    blackboard = Client(name='BehaviorTree')
    register_commands(blackboard, can_write=True)
    init_commands = {k: None for k in Commands.all()}
    write_variables(blackboard, Namespace.COMMANDS, init_commands)
    return tree


def get_has_get_stone_pickaxe(name, quantity):
    root = Selector(name=name + '_root')
    precondition = behaviors.HasItem(name=name + '_precondition', item=Items.STONE_PICKAXE,
                                     quantity=quantity)
    sequence = MemorylessSequence(name=name + '_sequence')

    root.add_children([precondition, sequence])

    get_wood_pickaxe = get_has_get_wood_pickaxe(name=name + '_has_get_wood', quantity=1)
    get_wood = get_has_get_item_subtree(name=name + '_get_wood', item_block=Items.LOG,
                                        item_floating=Items.LOG, quantity=1)
    get_cobblestone = get_has_get_item_subtree(name + '_get_cobblestone',
                                               item_block=Items.STONE,
                                               item_floating=Items.COBBLESTONE,
                                               quantity=3)
    craft = get_craft_stone_pickaxe(name=name + '_craft_stone_pickaxe')
    sequence.add_children([get_wood_pickaxe, get_wood, get_cobblestone, craft])
    return root


def get_craft_stone_pickaxe(name):
    root = MemorylessSequence(name=name + '_root')

    # Get sticks subtree
    sticks_root = Selector(name=name + '_sticks_root')
    has_sticks = behaviors.HasItem(name=name + '_has_sticks', item=Items.STICK, quantity=2)
    get_sticks_sequence = MemorylessSequence(name=name + '_sticks_sequence')
    has_craft_planks = get_has_craft_item_subtree(name=name + '_craft_planks', item=Items.PLANKS,
                                                  quantity=2)
    craft_sticks = get_has_craft_item_subtree(name=name + '_craft_sticks', item=Items.STICK,
                                              quantity=2)
    get_sticks_sequence.add_children([has_craft_planks, craft_sticks])
    sticks_root.add_children([has_sticks, get_sticks_sequence])

    # Get cobblestone subtree
    has_cobblestone = behaviors.HasItem(name=name + '_has_cobblestone', item=Items.COBBLESTONE,
                                        quantity=3)

    # Finally, craft the pickaxe
    craft_pickaxe = behaviors.Craft(name=name + '_craft_stone_pickaxe', item=Items.STONE_PICKAXE)

    root.add_children([sticks_root, has_cobblestone, craft_pickaxe])
    return root


def get_has_get_wood_pickaxe(name, quantity):
    root = Selector(name=name + '_root')
    has_pickaxe = behaviors.HasItem(name=name + '_has_wooden_pickaxe', item=Items.WOODEN_PICKAXE,
                                    quantity=quantity)
    sequence = MemorylessSequence(name=name + '_craft_pickaxe_sequence')
    get_wood = get_has_get_item_subtree(name=name + '_get_wood',
                                        item_floating=Items.LOG,
                                        item_block=Items.LOG,
                                        quantity=3)
    craft_pickaxe = get_craft_wood_pickaxe_subtree(name=name + '_craft_wood_pickaxe')

    sequence.add_children([get_wood, craft_pickaxe])
    root.add_children([has_pickaxe, sequence])
    return root


def get_craft_wood_pickaxe_subtree(name):
    root = MemorylessSequence(name=name + '_root')

    # Has/Get sticks subtree
    sticks_root = Selector(name=name + '_sticks_root')
    has_sticks = behaviors.HasItem(name=name + '_has_sticks', item=Items.STICK, quantity=2)
    craft_sticks_sequence = MemorylessSequence(name=name + '_craft_sticks_root')
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


def get_has_get_item_subtree(name, item_block, item_floating, quantity):
    root = Selector(name=name + '_root')
    # Floating items always have the name they would have in inventory
    has_item = behaviors.HasItem(name=name + '_precondition', item=item_floating, quantity=quantity)
    sequence = MemorylessSequence(name=name + '_sequence')
    is_get_close_to = get_is_get_close_to_subtree(name=name + '_is_get_close_to_',
                                                  item_floating=item_floating,
                                                  item_block=item_block,
                                                  tolerance_floating=CLOSE_TO_FLOATING,
                                                  tolerance_block=CLOSE_TO_BLOCK,
                                                  comm_variable=Commands.HAS_GET_ITEM)
    destroy = behaviors.Destroy(name=name + '_destroy', comm_variable=Commands.HAS_GET_ITEM)

    sequence.add_children([is_get_close_to, destroy])
    root.add_children([has_item, sequence])
    return root


def get_is_get_close_to_subtree(name, item_block, item_floating, comm_variable,
                                tolerance_block=CLOSE_TO_BLOCK,
                                tolerance_floating=CLOSE_TO_FLOATING):
    root = Selector(name + '_root')
    # Precondition
    precond_select = Selector(name=name + '_preconditions_root')
    is_close_floating = behaviors.IsCloseToEntity(name + '_precondition_floating',
                                                  entity=item_floating,
                                                  tolerance=tolerance_floating,
                                                  out_variable=comm_variable)
    is_close_block = behaviors.IsCloseToBlock(name=name + '_precondition_block',
                                              item=item_block,
                                              tolerance=tolerance_block,
                                              out_variable=comm_variable)
    precond_select.add_children([is_close_floating, is_close_block])

    # Check if there is a floating entity of the specified item around
    move_to_entity_sequence = MemorylessSequence(name=name + '_move_to_entity_sequence')
    has_entity_nearby = behaviors.IsCloseToEntity(name + '_is_close_to_floating',
                                                  entity=item_floating,
                                                  out_variable=comm_variable,
                                                  tolerance=None)
    move_to_entity = behaviors.MoveTo(name=name + '_move_to',
                                      destination_key=comm_variable,
                                      tolerance=tolerance_floating)
    move_to_entity_sequence.add_children([has_entity_nearby, move_to_entity])

    # Check if there is the specified block item around
    move_to_item_sequence = MemorylessSequence(name=name + '_MoveTo_sequence')
    has_nearby_item = behaviors.IsCloseToBlock(name + '_has_nearby',
                                               item=item_block,
                                               tolerance=None,
                                               out_variable=comm_variable)
    move_to_item = behaviors.MoveTo(name=name + '_move_to_item',
                                    destination_key=comm_variable,
                                    tolerance=tolerance_block)
    move_to_item_sequence.add_children([has_nearby_item, move_to_item])

    root.add_children([precond_select, move_to_entity_sequence, move_to_item_sequence])
    return root
