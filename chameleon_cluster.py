import itertools
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import metis
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def knn_graph(df, k, verbose=False):
    points = [p[1:] for p in df.itertuples()]
    g = nx.Graph()
    for i in range(0, len(points)):
        g.add_node(i)
    if verbose:
        print("Building kNN graph (k = %d)..." % (k))
    iterpoints = tqdm(enumerate(points), total=len(
        points)) if verbose else enumerate(points)
    for i, p in iterpoints:
        distances = list(map(lambda x: euclidean_distance(p, x), points))
        closests = np.argsort(distances)[1:k+1]  # second trough kth closest
        # print(distances[0])
        for c in closests:
            g.add_edge(i, c, weight=1.0 / distances[c], similarity=int(
                1.0 / distances[c] * 1e4))
        g.node[i]['pos'] = p
    g.graph['edge_weight_attr'] = 'similarity'
    return g


def part_graph(graph, k, df=None):
    edgecuts, parts = metis.part_graph(
        graph, 2, objtype='cut', ufactor=250)
    # print(edgecuts)
    for i, p in enumerate(graph.nodes()):
        graph.node[p]['cluster'] = parts[i]
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph


def pre_part_graph(graph, k, df=None, verbose=False):
    if verbose:
        print("Begin clustering...")
    clusters = 0
    for i, p in enumerate(graph.nodes()):
        graph.node[p]['cluster'] = 0
    cnts = {}
    cnts[0] = len(graph.nodes())

    while clusters < k - 1:
        maxc = -1
        maxcnt = 0
        for key, val in cnts.items():
            if val > maxcnt:
                maxcnt = val
                maxc = key
        s_nodes = [n for n in graph.node if graph.node[n]['cluster'] == maxc]
        s_graph = graph.subgraph(s_nodes)
        edgecuts, parts = metis.part_graph(
            s_graph, 2, objtype='cut', ufactor=250)
        new_part_cnt = 0
        for i, p in enumerate(s_graph.nodes()):
            if parts[i] == 1:
                graph.node[p]['cluster'] = clusters + 1
                new_part_cnt = new_part_cnt + 1
        cnts[maxc] = cnts[maxc] - new_part_cnt
        cnts[clusters + 1] = new_part_cnt
        clusters = clusters + 1

    edgecuts, parts = metis.part_graph(graph, k)
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph


def get_cluster(graph, clusters):
    nodes = [n for n in graph.node if graph.node[n]['cluster'] in clusters]
    return nodes


def connecting_edges(partitions, graph):
    cut_set = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    return cut_set


def min_cut_bisector(graph):
    graph = graph.copy()
    graph = part_graph(graph, 2)
    partitions = get_cluster(graph, [0]), get_cluster(graph, [1])
    return connecting_edges(partitions, graph)


def get_weights(graph, edges):
    return [graph[edge[0]][edge[1]]['weight'] for edge in edges]


def bisection_weights(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = min_cut_bisector(cluster)
    weights = get_weights(cluster, edges)
    return weights

def internal_interconnectivity(graph, cluster):
    return np.sum(bisection_weights(graph, cluster))


def relative_interconnectivity(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = internal_interconnectivity(
        graph, cluster_i), internal_interconnectivity(graph, cluster_j)
    return EC / ((ECci + ECcj) / 2.0)


def internal_closeness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = get_weights(cluster, edges)
    return np.sum(weights)


def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(
        graph, cluster_i), internal_closeness(graph, cluster_j)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(
        bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge_score(g, ci, cj, a):
    return relative_interconnectivity(
        g, ci, cj) * np.power(relative_closeness(g, ci, cj), a)


def merge_best(graph, df, a, k, verbose=False):
    clusters = np.unique(df['cluster'])
    max_score = 0
    ci, cj = -1, -1
    if len(clusters) <= k:
        return False

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose:
                print("Checking c%d c%d" % (i, j))
            gi = get_cluster(graph, [i])
            gj = get_cluster(graph, [j])
            edges = connecting_edges(
                (gi, gj), graph)
            if not edges:
                continue
            ms = merge_score(graph, gi, gj, a)
            if verbose:
                print("Merge score: %f" % (ms))
            if ms > max_score:
                if verbose:
                    print("Better than: %f" % (max_score))
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose:
            print("Merging c%d and c%d" % (ci, cj))
        df.loc[df['cluster'] == cj, 'cluster'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.node[p]['cluster'] == cj:
                graph.node[p]['cluster'] = ci
    return max_score > 0


def cluster(df, k, knn=10, m=30, alpha=2.0, verbose=False):
    graph = knn_graph(df, knn, verbose=True)
    graph = pre_part_graph(graph, m, df, verbose=True)
    iterm = tqdm(enumerate(range(m - k)), total=m-k)
    for i in iterm:
        merge_best(graph, df, alpha, k, verbose)
    res = rebuild_labels(df)
    return res

def rebuild_labels(df):
    ans = df.copy()
    clusters = list(pd.DataFrame(df['cluster'].value_counts()).index)
    c = 1
    for i in clusters:
        ans.loc[df['cluster'] == i, 'cluster'] = c
        c = c + 1
    return ans