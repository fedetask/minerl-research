from Keys import Items
from malmo_env import MalmoEnv

RECIPES = {
    Items.PLANKS: {'out_quantity': 4, 'ingredients': [(1, Items.LOG)]},
    Items.CRAFTING_TABLE: {'out_quantity': 1, 'ingredients': [(4, Items.PLANKS)]},
    Items.STICK: {'out_quantity': 4, 'ingredients': [(2, Items.PLANKS)]},
    Items.WOODEN_PICKAXE: {'out_quantity': 1, 'ingredients': [(3, Items.PLANKS), (2, Items.STICK)]}
}


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
