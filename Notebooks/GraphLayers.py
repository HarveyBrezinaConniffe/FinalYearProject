import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import Input, Model
import tensorflow as tf
import numpy as np

symbol_to_int = {'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33, 'Br': 34, 'Kr': 35, 'Rb': 36, 'Sr': 37, 'Y': 38, 'Zr': 39, 'Nb': 40, 'Mo': 41, 'Tc': 42, 'Ru': 43, 'Rh': 44, 'Pd': 45, 'Ag': 46, 'Cd': 47, 'In': 48, 'Sn': 49, 'Sb': 50, 'Te': 51, 'I': 52, 'Xe': 53, 'Cs': 54, 'Ba': 55, 'La': 56, 'Ce': 57, 'Pr': 58, 'Nd': 59, 'Pm': 60, 'Sm': 61, 'Eu': 62, 'Gd': 63, 'Tb': 64, 'Dy': 65, 'Ho': 66, 'Er': 67, 'Tm': 68, 'Yb': 69, 'Lu': 70, 'Hf': 71, 'Ta': 72, 'W': 73, 'Re': 74, 'Os': 75, 'Ir': 76, 'Pt': 77, 'Au': 78, 'Hg': 79, 'Tl': 80, 'Pb': 81, 'Bi': 82, 'Po': 83, 'At': 84, 'Rn': 85, 'Fr': 86, 'Ra': 87, 'Ac': 88, 'Th': 89, 'Pa': 90, 'U': 91, 'Np': 92, 'Pu': 93, 'Am': 94, 'Cm': 95, 'Bk': 96, 'Cf': 97, 'Es': 98, 'Fm': 99, 'Md': 100, 'No': 101, 'Lr': 102, 'Rf': 103, 'Db': 104, 'Sg': 105, 'Bh': 106, 'Hs': 107, 'Mt': 108, 'Ds ': 109, 'Rg ': 110, 'Cn ': 111, 'Nh': 112, 'Fl': 113, 'Mc': 114, 'Lv': 115, 'Ts': 116, 'Og': 117}

def convertFromNetworkX(graph, maxNodes, maxEdges, embeddingDim):
    nodeEmbeddings = np.zeros((maxNodes, embeddingDim))
    edgeEmbeddings = np.zeros((maxEdges, embeddingDim))
    universalEmbedding = np.zeros((embeddingDim))
    adjacencyMatrix = np.zeros((maxNodes, maxNodes))
    connectedEdges = np.zeros((maxNodes, maxEdges))
    
    # Populate node embeddings.
    for nodeNum, symbol in graph.nodes(data="element"):
        symbolInt = symbol_to_int[symbol]
        nodeEmbeddings[nodeNum][symbolInt] = 1.0
        
    # Populate edge embeddings and adjacency matrix.
    i = 0
    for start, end in graph.edges:
        edgeOrder = graph.get_edge_data(start, end)["order"]

        # Kinda hacky, Edgeorder can be 1.5 prob should map not multiply
        edgeEmbeddings[i][int(edgeOrder*2)] = 1.0
        
        adjacencyMatrix[start][end] = 1.0
        adjacencyMatrix[end][start] = 1.0
        
        connectedEdges[start][i] = 1.0
        connectedEdges[end][i] = 1.0
        
        i += 1
    
    return nodeEmbeddings, edgeEmbeddings, universalEmbedding, adjacencyMatrix, connectedEdges

class GraphUpdate(keras.layers.Layer):
    def __init__(self, 
                 v_out_dim,
                 e_out_dim,
                 u_out_dim,
                 activation="relu"):
        super(GraphUpdate, self).__init__()
        self.v_update = Dense(v_out_dim, activation=activation, name="V_Update")
        self.e_update = Dense(e_out_dim, activation=activation, name="E_Update")
        self.u_update = Dense(u_out_dim, activation=activation, name="U_Update")

    def call(self, inputs):
        v_in, e_in, u_in, adj, conEd = inputs
        v_out = self.v_update(v_in)
        e_out = self.e_update(e_in)
        u_out = self.u_update(u_in)
        return [v_out, e_out, u_out, adj, conEd]

# Add the embedding of each connected edge to each vertex.
class PoolEdgesToVertices(keras.layers.Layer):
    def __init__(self):
        super(PoolEdgesToVertices, self).__init__()

    def call(self, inputs):
        v_in, e_in, u_in, adj, conEd = inputs

        pooledEdges = tf.matmul(conEd, e_in)
        v_out = v_in+pooledEdges

        return [v_out, e_in, u_in, adj, conEd]

# Add the embedding of each connected vertex to each edge.
class PoolVerticesToEdges(keras.layers.Layer):
    def __init__(self):
        super(PoolVerticesToEdges, self).__init__()

    def call(self, inputs):
        v_in, e_in, u_in, adj, conEd = inputs

        pooledNodes = tf.matmul(conEd, v_in, transpose_a=True)
        e_out = e_in+pooledNodes

        return [v_in, e_out, u_in, adj, conEd]

# Pool all vertices to universal.
class PoolVerticesToUniversal(keras.layers.Layer):
    def __init__(self):
        super(PoolVerticesToUniversal, self).__init__()

    def call(self, inputs):
        v_in, e_in, u_in, adj, conEd = inputs

        u_out = tf.reduce_sum(v_in, axis=-1)

        return [v_in, e_in, u_out, adj, conEd]

# Pool universal to all vertices.
class PoolUniversalToVertices(keras.layers.Layer):
    def __init__(self):
        super(PoolUniversalToVertices, self).__init__()

    def call(self, inputs):
        v_in, e_in, u_in, adj, conEd = inputs

        u_tiled = tf.tile(tf.expand_dims(u_in, axis=1), [1, v_in.shape[1], 1])
        v_out = v_in+u_tiled

        return [v_out, e_in, u_in, adj, conEd]

# Pool universal to all edges.
class PoolUniversalToEdges(keras.layers.Layer):
    def __init__(self):
        super(PoolUniversalToEdges, self).__init__()

    def call(self, inputs):
        v_in, e_in, u_in, adj, conEd = inputs

        u_tiled = tf.tile(tf.expand_dims(u_in, axis=1), [1, e_in.shape[1], 1])
        e_out = e_in+u_tiled

        return [v_in, e_out, u_in, adj, conEd]