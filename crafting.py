from Keys import Items
from malmo_env import MalmoEnv

RECIPES = {
    Items.PLANKS: {'out_quantity': 4, 'ingredients': [(1, Items.LOG)]},
    Items.CRAFTING_TABLE: {'out_quantity': 1, 'ingredients': [(4, Items.PLANKS)]},
    Items.STICK: {'out_quantity': 4, 'ingredients': [(2, Items.PLANKS)]},
    Items.WOODEN_PICKAXE: {'out_quantity': 1, 'ingredients': [(3, Items.PLANKS), (2, Items.STICK)]},
    Items.STONE_PICKAXE: {'out_quantity': 1, 'ingredients': [(2, Items.STICK), (3, Items.COBBLESTONE)]}
}

# Maps the item that one wants to destroy with a list of tools that can be used to destroy it
DESTROY_TOOLS = {Items.LOG: [],
                 Items.STONE: [Items.STONE_PICKAXE, Items.WOODEN_PICKAXE]}


def can_craft(item, inventory) -> bool:
    """Check whether the given item can be crafted from the inventory.

    Args:
        item (str): Item to craft. Must be one of Keys.Items
        inventory (dict): An inventory observation

    Returns:
        True if the given item can be crafted, False otherwise

    """
    recipe = RECIPES[item]
    for quantity, ingredient in recipe['ingredients']:
        # Check that inventory contains the given
        available_items = MalmoEnv.get_item_from_inventory(inventory, ingredient)
        if not available_items:
            return False
        tot_quantity = sum(item['quantity'] for item in available_items)
        if tot_quantity < quantity:
            return False
    return True


def get_tool_to_destroy(block):
    """Get the tool to destroy the given block.

    Args:
        block (str): Name of the block that has to be destroyed.

    Returns:
        A list of tools that can destroy the block. An empty list means that it can be destroyed
        with bare hands.

    """
    if block in DESTROY_TOOLS:
        return DESTROY_TOOLS[block]
    else:
        return []
