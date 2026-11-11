import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__ (self, index):
        self.index = index
        self.neighbours = []

    def add_neighbour(self, other_node, transition_prob):
        self.neighbours.append((other_node, transition_prob))

    def draw_node(self):
        nodes = [node for node, _ in self.neighbours]
        probs = [p for _, p in self.neighbours]
        i = np.random.choice(len(nodes), p=probs)
        return nodes[i]

class System:
    def __init__(self, nodes, initial_state=0):
        assert len(nodes) > initial_state, "Initial states exceed no. of states"
        self.nodes = nodes
        self.transition_matrix()
        self.prob_vector = np.zeros(len(nodes), dtype=float)
        self.prob_vector[initial_state] = 1
        self.current_node = nodes[initial_state]
        self.history = [initial_state]

    def transition_matrix(self):
        dimension = len(self.nodes)
        self.Transition = np.zeros((dimension, dimension))
        for node in self.nodes:
            for neighbour_node, t in node.neighbours:
                self.Transition[node.index, neighbour_node.index] = t

    def propagate_pvector(self, n):
        for _ in range(n):
            self.prob_vector = self.prob_vector @ self.Transition

    @property
    def state(self):
            return self.current_node.index
    
    def step(self):
        self.current_node = self.current_node.draw_node()
        self.history.append(self.current_node.index)
        return self.state
    
    def simulate(self, n):
        for _ in range(n):
            self.step()
        return self.history


def build_nodes(transition_probs, n=3):
    nodes = [Node(i) for i in range(n)]
    for i, j, t in transition_probs:
        nodes[i].add_neighbour(nodes[j], t)
    return nodes

    
    