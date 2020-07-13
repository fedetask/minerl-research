from py_trees.common import Access
from py_trees.blackboard import Client
from enum import Enum


class Namespace(Enum):
    OBSERVATIONS = 'observations'
    ACTIONS = 'actions'
    COMMANDS = 'commands'


def register_variables(client, namespace, can_write, variables):
    """Register the given client to the given variables in the given blackboard.

    Args:
        client (Client): The client to register on the given variables.
        namespace (Namespace): The namespace under which the given variables will be registered. Leave
        empty for the default namespace / or if the variables already contain the namespace as
        prefix.
        can_write (bool): Whether to register the client in read-only or read-write mode.
        variables (list): List with the names of the variables to register

    """
    access = Access.WRITE if can_write else Access.READ
    for var in variables:
        client.register_key(key=namespace.value + '/' + var, access=access)


def write_variables(client, namespace, variables):
    """Write the given key-value pairs in the blackboard under the given namespace.

    Args:
        client (Client): The client that is writing the variables.
        namespace (Namespace): The namespace under which the variables are registered. Leave empty
        for default namespace / or if variables already contain the namespace as prefix.
        variables (dict): Dict of key-value to write, where the keys must be already registered.

    """
    for k, v in variables.items():
        setattr(client, namespace.value + '/' + k, v)


def read_variables(client, namespace, variables):
    """Read the given variables from the blackboard and return them as a dictionary.

    Args:
        client (Client): The client that is reading the variables.
        namespace (Namespace): The name space from which to read the variables
        variables (list): List containing the names names of the variables to read.

    Returns:
        A dictionary with the variables.

    """
    variables = {var: client.get(namespace.value + '/' + var) for var in variables}
    return variables
