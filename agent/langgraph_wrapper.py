"""Provides a shim for LangGraph. Tries to use the real package if installed,
otherwise provides a minimal compatible version for the project to run.
"""
try:
    import langgraph as _lg
    Graph = _lg.Graph
    State = _lg.State
except Exception:
    # Minimal shim
    class State:
        def __init__(self):
            self.store = {}

        def set(self, key, value):
            self.store[key] = value

        def get(self, key, default=None):
            return self.store.get(key, default)

    class Graph:
        def __init__(self):
            self._nodes = {}

        def add_node(self, name, func):
            self._nodes[name] = func

        def run(self, node_name, *args, **kwargs):
            if node_name not in self._nodes:
                raise KeyError(f"Node {node_name} not found")
            return self._nodes[node_name](*args, **kwargs)

        # compatibility helper
        def register(self, name, func):
            self.add_node(name, func)
