import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from numba import autojit
import random
import heapq
import sys

def random_vector(min_max_pairs):
    v = []
    for (min_val, max_val) in min_max_pairs:
        v.append(random.uniform(min_val, max_val))
    
    return np.array(v)

# Returns [w, C, tc]
#def trn(X, t_max, N, epsilon_i, epsilon_f, Ti, Tf, lambda_i, lambda_f):
def trn(data_set, max_iterations, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f):
    connections = np.empty((codebook_size, codebook_size), dtype=object)
    # generate codebook vectors
    codebook = []
    for i in range(codebook_size):
        codebook.append(
            random_vector([
                (0, 30),
                (0, 30),
                (0, 30)]))

    for t in range(max_iterations):
        # Progress bar
        amtDone = float(t) / max_iterations * 100
        sys.stdout.write("\r%.2f%%" %amtDone)
        sys.stdout.flush()

        # Select random data point
        random_data_point = data_set[random.randint(0, len(data_set)-1)]
        
        codebooks_near_point = np.zeros(codebook_size)
            
        # for each w[i] find the number of w[j] such that || x-w[j] || < || x - w[i] ||
        # array of all x - w[i]
        # array of all || x - w[i] ||
        distances = []

        for i in range(codebook_size):
            distances.append(np.linalg.norm(random_data_point - codebook[i]))
        
        lambda_val = lambda_i * ((lambda_f / lambda_i) ** (float(t) / max_iterations));
        epsilon = epsilon_i * ((epsilon_f / epsilon_i) ** (float(t) / max_iterations));
        for i in range(codebook_size):
            codebooks_near_point[i] = np.sum(distances < distances[i])
            # update the codebook vectors according to the neural gas algorithm
            #for i in range(0, codebook_size):
            codebook[i] = codebook[i] + epsilon * np.exp(-1 * codebooks_near_point[i] / lambda_val) * (random_data_point - codebook[i])

        # find closest two codebook indices <- SOMETHING WRONG HERE
        smallest1, smallest2 = heapq.nsmallest(2, [(v, i) for (i, v) in enumerate(distances)])
        index_smallest1, index_smallest2 = smallest1[1], smallest2[1]
        # create connection and refresh existing connection
        connections[index_smallest1, index_smallest2] = [1, 0]
        connections[index_smallest2, index_smallest1] = [1, 0]
            
            
        # age all connections <- THIS IS TERRIBLE
        max_age = T_i * ((T_f / T_i) ** (float(t) / max_iterations));
        for row in range(len(connections)):
            for col in range(len(connections[row])):
                if connections[row][col] != None:
                    connections[row][col][1] += 1
                    if connections[row][col][1] > max_age:
                        connections[row][col] = None

    return codebook, connections

def connections_to_graph(connections):
    adjacency_matrix = np.zeros((len(connections), len(connections)))

    for i in range(len(connections)):
        for j in range(len(connections[i])):
            if connections[i,j]:
                adjacency_matrix[i,j] = 1
            else:
                adjacency_matrix[i,j] = 0

    G = nx.Graph(adjacency_matrix)
    return G

def draw_graph(G, file_name):
    nx.draw(G)
    plt.savefig(file_name)
            
def output_json(G):
    import json
    f = open("graph.json", "w")
    data = json_graph.node_link_data(G)
    serialized_data = json.dumps(data)
    f.write(serialized_data)
    f.close()


def main():
    dataset = np.genfromtxt("swissroll.csv", delimiter="\t")

    num_points = len(dataset)
    import time
    t0 = time.time()
    epsilon_i = 0.3
    epsilon_f = 0.05
    lambda_i = 0.2 * num_points
    lambda_f = 0.01
    T_i = 0.1 * num_points
    T_f = 0.5 * num_points
    codebook_size = 200
    max_iter = 200 * codebook_size
    print "Constructing network on", len(dataset), "data points..."
    codebook, connections = trn(dataset, max_iter, codebook_size, epsilon_i, epsilon_f, lambda_i, lambda_f, T_i, T_f)
    print "TRN Runtime:", time.time() - t0

    print "\nCodebook:"
    for item in codebook:
        print item[0],",", item[1],",",item[2]

    print "\nConnections:"

    G = connections_to_graph(connections)
    
    print "Drawing graph..."
    print G.edges()
    draw_graph(G, "graph.png")
    output_json(G)
    print "Done!"
    #output_json(codebook, connections)

if __name__ == "__main__":
    main()

