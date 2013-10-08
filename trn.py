import numpy as np
import numpy.matlib 
from numba import autojit
import random
import heapq

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
    for i in range(0, codebook_size):
        codebook.append(
            random_vector([
                (0, 10),
                (0, 10),
                (0, 10),
                (0, 10),
                (0, 10)]))

    for __data_point__ in range(0, len(data_set)):
        random_data_point = data_set[random.randint(0, len(data_set) - 1)] # choose a random data point

        for t in range(0, max_iterations):
            codebooks_near_point = np.zeros(codebook_size)
            
            # for each w[i] find the number of w[j] such that || x-w[j] || < || x - w[i] ||
            # array of all x - w[i]
            # array of all || x - w[i] ||
            """
            for i in range(0, codebook_size):
            for j in range(0, codebook_size):
            if j == i:
            pass
            else:
            if np.linalg.norm(random_data_point - codebook[j]) < np.linalg.norm(random_data_point - codebook[i]):
            codebooks_near_point[i] += 1
            """
            distances = []

            for i in range(0, codebook_size):
                distances.append(np.linalg.norm(random_data_point - codebook[i]))
        
            for i in range(0, codebook_size):
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
            for row in range(0, len(connections)):
                for col in range(0, len(connections[row])):
                    if connections[row][col] != None:
                        connections[row][col][1] += 1
                        if connections[row][col][1] > max_age:
                            connections[row][col] = None

    return codebook, connections


def output_json(codebook, connections):
    f = open("graph.json", "w")
    for i in range(0, len(connections)):
        for j in range(0, len(connections[i])):
            if connections[i][j] != None:
                f.write("{'source':"+str(i)+", 'target':"+str(j)+", 'value':1},\n") 
    f.close()


def main():
    num_points = 200
    dataset = []
    for i in range(0, num_points):
        dataset.append(random_vector([
            (1,10),
            (1,10),
            (1,10),
            (1,10),
            (1,10)]))

    import time
    t0 = time.time()
    max_iter = int(0.2 * num_points)
    epsilon = 0.2
    lambda_i = 0.5
    max_age = int(0.9 * num_points)
    codebook_size = 20
    print "Constructing network on", len(dataset), "data points..."
    codebook, connections = trn(dataset, max_iter, codebook_size, epsilon, lambda_i, max_age)
    print "TRN Runtime:", time.time() - t0
    #output_json(codebook, connections)

if __name__ == "__main__":
    main()

