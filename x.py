import networkx as nx
import matplotlib.pyplot as plt

# Initialize a directed graph to represent the roadmap
G = nx.DiGraph()

# Step 1: Add all nodes (components from the Padlet)

# Assessments and Score
G.add_node("18 Questions Assessment")
G.add_node("Score Card")
G.add_node("Follow-up Questions")

# Connection Questions (Both Paper-Based and Digital)
G.add_node("Connection Questions (Paper-Based)")
G.add_node("Connection Questions (Digital)")

# Purpose and Energy Questions
G.add_node("Purpose Questions")
G.add_node("Energy Questions")

# States: Fatigue, Indulgent, Maximized, Reserved
G.add_node("Depleted")
G.add_node("Fatigued")
G.add_node("Indulgent")
G.add_node("Maximized")
G.add_node("Sustained")
G.add_node("Reserved")

# Tools and Resources
G.add_node("Load Management")
G.add_node("Capacity Net Worth")
G.add_node("Value Mining")
G.add_node("CHIEFF")

#depleted stuff
G.add_node("Functioning Burnout Checklist")



# Value Mining Components
G.add_node("Value Mining Facilitator")
G.add_node("Value Mining Values List")
G.add_node("Value Mining - User")
G.add_node("Value Mining Exercise")

# Personal Value and Egocake
G.add_node("Personal Value Proposition")
G.add_node("Egocake")

# Reflection and Load Management Follow-up
G.add_node("Reflection Questions")
G.add_node("Load Management Follow-up")
G.add_node("Load Management TTT")
G.add_node("TTT Work Book")

# 3x3 Reserved and other states
G.add_node("3x3 Capacity Matrix")
G.add_node("3x3 Reserved")
G.add_node("3x3 Indulgent")
G.add_node("3x3 Fatigued")
G.add_node("3x3 Maximized")
G.add_node("Healthy Hustle")
G.add_node("Maintenance")

# Capacity coaching scripts
G.add_node("Capacity coaching scripts")
G.add_node("Coaching script Maximized")
G.add_node("Coaching script Indulgent")
G.add_node("Coaching script Fatigued")
G.add_node("Coaching script reserved")

G.add_node("Resignation")

# Step 2: Add edges based on the whiteboard components with sequence and duration attributes

# States to Tools with sequence and duration
G.add_edge("Maximized", "3x3 Fatigued", **{"sequence": 4, "duration": 14})
G.add_edge("Maximized", "Personal Value Proposition", **{"sequence": 1, "duration": 14})
G.add_edge("Maximized", "CHIEFF", **{"sequence": 2, "duration": 14})
G.add_edge("Maximized", "Ego Cake", **{"sequence": 3, "duration": 14})
G.add_edge("Maximized", "Resignation", **{"sequence": 5, "duration": 14})

G.add_edge("Sustained", "3x3 Capacity Matrix", **{"sequence": 2, "duration": 14})
G.add_edge("Sustained", "Personal Value Proposition", **{"sequence": 3, "duration": 14})
G.add_edge("Sustained", "CHIEFF", **{"sequence": 1, "duration": 14})
G.add_edge("Sustained", "Capacity Net Worth", **{"sequence": 4, "duration": 14})

G.add_edge("Indulgent", "Value Mining", **{"sequence": 2, "duration": 14})
G.add_edge("Indulgent", "Personal Value Proposition", **{"sequence": 1, "duration": 14})
G.add_edge("Indulgent", "Resignation", **{"sequence": 4, "duration": 14})
G.add_edge("Indulgent", "Coaching script Indulgent", **{"sequence": 3, "duration": 14})

G.add_edge("Fatigued", "Load Management", **{"sequence": 1, "duration": 14})
G.add_edge("Fatigued", "Ego Cake", **{"sequence": 4, "duration": 14})
G.add_edge("Fatigued", "3x3 Fatigued", **{"sequence": 3, "duration": 14})
G.add_edge("Fatigued", "Capacity Net Worth", **{"sequence": 2, "duration": 14})

G.add_edge("Reserved", "CHIEFF", **{"sequence": 1, "duration": 14})
G.add_edge("Reserved", "Value Mining", **{"sequence": 3, "duration": 14})
G.add_edge("Reserved", "Ego Cake", **{"sequence": 4, "duration": 14})
G.add_edge("Reserved", "Coaching script reserved", **{"sequence": 2, "duration": 14})

G.add_edge("Depleted", "Functioning Burnout Checklist", **{"sequence": 1, "duration": 14})
G.add_edge("Depleted", "CHIEFF", **{"sequence": 2, "duration": 14})
G.add_edge("Depleted", "Load Management", **{"sequence": 3, "duration": 14})
G.add_edge("Depleted", "Coaching script Fatigued", **{"sequence": 4, "duration": 14})

# Load Management and TTT
G.add_edge("Load Management", "Load Management Follow-up", **{"sequence": 1, "duration": 14})
G.add_edge("Load Management", "Load Management TTT", **{"sequence": 2, "duration": 14})
G.add_edge("Load Management TTT", "Load Management Follow-up", **{"sequence": 3, "duration": 14})
G.add_edge("Load Management Follow-up", "Reflection Questions", **{"sequence": 4, "duration": 14})

# Value Mining Components Connections
G.add_edge("Value Mining", "Value Mining Facilitator", **{"sequence": 1, "duration": 14})
G.add_edge("Value Mining", "Value Mining Values List", **{"sequence": 2, "duration": 14})
G.add_edge("Value Mining", "Value Mining Exercise", **{"sequence": 3, "duration": 14})
G.add_edge("Value Mining Values List", "Value Mining - User", **{"sequence": 4, "duration": 14})
G.add_edge("Value Mining Exercise", "Value Mining - User", **{"sequence": 5, "duration": 14})

# Personal Value and Egocake Links
G.add_edge("Personal Value Proposition", "Egocake", **{"sequence": 1, "duration": 14})

# 3x3 Connections
G.add_edge("3x3(energy def. purpose def.)", "3x3 Fatigued", **{"sequence": 1, "duration": 14})
G.add_edge("3x3(energy def. purpose def.)", "3x3 Indulgent", **{"sequence": 2, "duration": 14})
G.add_edge("3x3(energy def. purpose def.)", "3x3 Maximized", **{"sequence": 3, "duration": 14})
G.add_edge("3x3(energy def. purpose def.)", "3x3 Reserved", **{"sequence": 4, "duration": 14})

# Miscellaneous Connections
G.add_edge("Healthy Hustle", "Maintenance", **{"sequence": 1, "duration": 14})
G.add_edge("Reflection Questions", "Follow-up Questions", **{"sequence": 2, "duration": 14})
G.add_edge("Reserved", "Capacity Net Worth", **{"sequence": 3, "duration": 14})

# Coaching scripts connections
G.add_edge("Capacity coaching scripts", "Coaching script Maximized", **{"sequence": 1, "duration": 14})
G.add_edge("Capacity coaching scripts", "Coaching script Indulgent", **{"sequence": 2, "duration": 14})
G.add_edge("Capacity coaching scripts", "Coaching script Fatigued", **{"sequence": 3, "duration": 14})
G.add_edge("Capacity coaching scripts", "Coaching script reserved", **{"sequence": 4, "duration": 14})

def get_graph():
    import networkx as nx
    return G

# Optional: Visualization (if needed)
'''
def visualize_graph(G):
    plt.figure(figsize=(20, 15))  # Set the figure size
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", 
            font_size=8, font_weight="bold", arrows=True, 
            arrowstyle="->", arrowsize=20)
    plt.title("Complete Roadmap Graph Structure (All Components)")

    # Show the plot
    plt.show(block=True)  # Ensure the plot stays open

# Get the graph and visualize it
G = get_graph()
visualize_graph(G)
'''