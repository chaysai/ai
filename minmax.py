import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def is_terminal(self):
        return not bool(self.children)

    def evaluate(self):
        return self.value

    def get_children(self):
        return self.children


def minimax(node, depth, maximizing_player):
    if depth == 0 or node.is_terminal():
        return node.evaluate()

    if maximizing_player:
        max_eval = float('-inf')
        for child in node.get_children():
            eval = minimax(child, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.get_children():
            eval = minimax(child, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval


def alpha_beta(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or node.is_terminal():
        return node.evaluate()

    if maximizing_player:
        max_eval = float('-inf')
        for child in node.get_children():
            eval = alpha_beta(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.get_children():
            eval = alpha_beta(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def draw_tree_with_labels(node, x, y, dx):
    if node is not None:
        plt.text(x, y, str(node.label), fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))

        if len(node.children) > 0:
            next_dx = dx / len(node.children)
            next_x = x - dx/2 + next_dx/2
            next_y = y - 1

            for child in node.children:
                plt.plot([x, next_x], [y, next_y], 'k-', lw=1)
                draw_tree_with_labels(child, next_x, next_y, next_dx)
                next_x += next_dx


def visualize_evaluation(node, algorithm):
    if algorithm.__name__ == "minimax":
        evaluation_function = minimax
    elif algorithm.__name__ == "alpha_beta":
        evaluation_function = alpha_beta
    else:
        raise ValueError("Invalid algorithm specified")

    def evaluate(node, depth, maximizing_player):
        if evaluation_function.__name__ == "minimax":
            label = f"{node.value}\n{evaluation_function(node, depth, maximizing_player)}"
        else:
            label = f"{node.value}\n{evaluation_function(node, depth, float('-inf'), float('inf'), maximizing_player)}"
        node.label = label

        for child in node.children:
            evaluate(child, depth - 1, not maximizing_player)

    evaluate(node, 3, False)

    plt.figure(figsize=(10, 8))
    draw_tree_with_labels(node, 0, 0, 2)  # Use modified draw_tree function
    plt.axis('off')
    plt.show()



# Create the tree
root = Node(0)
child1 = Node(0)
child2 = Node(0)
child3 = Node(0)
child11 = Node(3)
child12 = Node(4)
child13 = Node(2)
child21 = Node(2)
child22 = Node(10)
child23 = Node(4)
child31 = Node(3)
child32 = Node(1)
child33 = Node(4)

root.add_child(child1)
root.add_child(child2)
root.add_child(child3)
child1.add_child(child11)
child1.add_child(child12)
child1.add_child(child13)
child2.add_child(child21)
child2.add_child(child22)
child2.add_child(child23)
child3.add_child(child31)
child3.add_child(child32)
child3.add_child(child33)
# Evaluate using Minimax
minimax_result = minimax(root, 3, True)
print("Minimax Result:", minimax_result)

# Evaluate using Alpha-Beta Pruning
alpha_beta_result = alpha_beta(root, 3, float('-inf'), float('inf'), True)
print("Alpha-Beta Pruning Result:", alpha_beta_result)
# Visualize Minimax
visualize_evaluation(root, minimax)

# Visualize Alpha-Beta Pruning
visualize_evaluation(root, alpha_beta)