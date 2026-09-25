import pandas as pd
import csv

from IPython.display import display

records = []    #create an empty array to store the purchase records of customers
idMap = {}      #initialize an empty set to store food names and ids

with open("categories.csv") as foodIDs:
    food = csv.reader(foodIDs)          #open the ID file to be read

    for line in food:
        id = int(line[0].strip())       #take the ID from the first entry of the file
        idMap[id] = line[1].strip()     #record the foods name in the idMap, related to its food ID


with open("data.csv", "r") as file:
    data = csv.reader(file)         #open the data to be read

    for line in data:
        time = line[0]          #records the time spent for each customer
        customer = line[1].strip()      #records customer ID, removes whitespace
        basket = []         #initialize an empty an array to store the food bought

        for cell in line[2:]:   #reads the rest of the cells in the row
            item = int(cell.strip())    #converts the cell entry to an integer
            if item in idMap:           #checks if the item is in the ID list
                basket.append(idMap[item])      #adds the name of the item to the customer's purchases

        records.append({            #puts the formatted data into the records array
            "Time": time,
            "customer_ID": customer,
            "items": basket
        })

dataframe = pd.DataFrame(records)

display(dataframe)




from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = dataframe["items"].tolist()      #the next few lines change the data into a list so
te = TransactionEncoder()                       #that it can be used in the apriori function, following
te_array = te.fit(transactions).transform(transactions)     #the documentation at geeksforgeeks.com
df_encoded = pd.DataFrame(te_array, columns = te.columns_)

display(df_encoded)



frequent_itemsets = apriori(df_encoded, min_support = .3, use_colnames=True)
print("Total Frequent Itemsets", frequent_itemsets.shape[0])
display(frequent_itemsets   )

# --- Build association rules and graph of co-purchases ---
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

import networkx as nx
import matplotlib.pyplot as plt

# We will create an undirected graph where items (singletons) are nodes
# and edges connect items that frequently appear together. Use confidence
# or lift as the edge weight depending on preference.
G = nx.Graph()

# Ensure we only add pairwise relations (single-item antecedents and consequents)
for _, row in rules.iterrows():
    antecedents = list(row['antecedents'])
    consequents = list(row['consequents'])
    # only consider rules that connect single items on each side
    if len(antecedents) == 1 and len(consequents) == 1:
        a = antecedents[0]
        b = consequents[0]
        weight = float(row.get('confidence', row.get('lift', 1.0)))
        # add nodes (labels are item names)
        G.add_node(a)
        G.add_node(b)
        # if edge exists, keep the max weight
        if G.has_edge(a, b):
            if G[a][b].get('weight', 0) < weight:
                G[a][b]['weight'] = weight
        else:
            G.add_edge(a, b, weight=weight)

if len(G) == 0:
    print("No pairwise single-item rules found with the current thresholds.")
else:
    # draw the graph
    plt.figure(figsize=(12, 9))
    # Try multiple layouts to reduce overlap:
    # 1) Graphviz 'neato' if available (often gives good spacing),
    # 2) Kamada-Kawai layout, 3) stronger spring layout fallback
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    except Exception:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, k=1.2, iterations=500, seed=42)

    # node sizes scaled by degree
    degrees = dict(G.degree())
    node_sizes = [200 + 1500 * (degrees[n] / max(1, max(degrees.values()))) for n in G.nodes()]

    # edge widths scaled by weight
    weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    edge_widths = [1 + 4 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    # offset labels slightly upward to reduce overlap with nodes
    label_pos = {n: (x, y + 0.03) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8)

    plt.title('Grocery co-purchase graph (edges weighted by rule confidence)')
    plt.axis('off')
    out_path = "co_purchase_graph.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved graph to {out_path}")
    plt.show()