import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
import networkx as nx
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
def trn(data_set, max_iterations, codebook_size, epsilon, lambda_val, max_age):
    connections = np.empty((codebook_size, codebook_size), dtype=object)
    # generate codebook vectors
    codebook = []
    for i in range(codebook_size):
        codebook.append(
            random_vector([
                (0, 30),
                (0, 30),
                (0, 30)]))

    for __data_point__ in range(len(data_set)):
        # Progress bar
        amtDone = float(__data_point__) / len(data_set) * 100
        sys.stdout.write("\r%.2f%%" %amtDone)
        sys.stdout.flush()

        random_data_point = data_set[random.randint(0, len(data_set) - 1)] # choose a random data point

        for t in range(max_iterations):
            codebooks_near_point = np.zeros(codebook_size)
            
            # for each w[i] find the number of w[j] such that || x-w[j] || < || x - w[i] ||
            # array of all x - w[i]
            # array of all || x - w[i] ||
            distances = []

            for i in range(codebook_size):
                distances.append(np.linalg.norm(random_data_point - codebook[i]))
        
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
            
def output_json(codebook, connections):
    f = open("graph.json", "w")
    for i in range(len(connections)):
        for j in range(len(connections[i])):
            if connections[i][j] != None:
                f.write("{'source':"+str(i)+", 'target':"+str(j)+", 'value':1},\n") 
    f.close()


def main():
    dataset = np.genfromtxt("swissroll.csv", delimiter="\t")

    num_points = len(dataset)
    import time
    t0 = time.time()
    max_iter = 50
    epsilon = 0.2
    lambda_i = 0.5
    max_age = 1000
    codebook_size = 50
    print "Constructing network on", len(dataset), "data points..."
    codebook, connections = trn(dataset, max_iter, codebook_size, epsilon, lambda_i, max_age)
    print "TRN Runtime:", time.time() - t0

    print "\nDataset:"
    for line in dataset:
        print line[0],",", line[1]

    print "\nCodebook:"
    for item in codebook:
        print item[0],",", item[1]

    print "\nConnections:"

    G = connections_to_graph(connections)
    
    print "Drawing graph..."
    print G.edges()
    draw_graph(G, "graph.png")
    print "Done!"
    #output_json(codebook, connections)

if __name__ == "__main__":
    main()

