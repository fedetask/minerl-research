from py_trees.common import Access
from py_trees.blackboard import Client
from enum import Enum
import Keys


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


def read_var(client, namespace, var_name):
    return client.get(namespace.value + '/' + var_name)


def set_var(client, namespace, var_name, var_value):
    setattr(client, namespace.value + '/' + var_name, var_value)


def read_command(client, key):
    return read_var(client, Namespace.COMMANDS, key)


def set_command(client, key, value):
    set_var(client, Namespace.COMMANDS, key, value)


def read_observation(client, key):
    return read_var(client, Namespace.OBSERVATIONS, key)


def set_observation(client, key, value):
    return set_var(client, Namespace.OBSERVATIONS, key, value)


def read_action(client, key):
    return read_var(client, Namespace.ACTIONS, key)


def set_action(client, key, value):
    set_var(client, Namespace.ACTIONS, key, value)


def register_observations(client, can_write, variables=None):
    to_register = variables if variables is not None else Keys.Observations.all()
    register_variables(client=client, namespace=Namespace.OBSERVATIONS, can_write=can_write,
                       variables=to_register)


def register_actions(client, can_write, variables=None):
    to_register = variables if variables is not None else Keys.Actions.all()
    register_variables(client=client, namespace=Namespace.ACTIONS, can_write=can_write,
                       variables=to_register)


def register_commands(client, can_write, variables=None):
    to_register = variables if variables is not None else Keys.Commands.all()
    register_variables(client=client, namespace=Namespace.COMMANDS, can_write=can_write,
                       variables=to_register)
