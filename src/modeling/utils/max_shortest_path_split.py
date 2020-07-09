

def max_shortest_path_split():


def get_negatives(x, pos_edge_index, mult):

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=len(list(set(pos_edge_index.cpu().numpy().flatten()))),
        num_neg_samples=pos_edge_index.size(1)*mult)

    map_dict = {x:i for i,x in enumerate(set(pos_edge_index.cpu().numpy().flatten()))}

    f = np.vectorize(lambda x: map_dict[x])
    pos_edge = f(pos_edge_index.cpu().numpy())

    G = nx.Graph()
    G.add_nodes_from(pos_edge.flatten())
    G.add_edges_from(pos_edge.T)

    pos = np.array([total_neighbors(G, x[0], x[1]) for x in pos_edge.T])
    neg = np.array([total_neighbors(G, x[0], x[1]) for x in neg_edge_index.cpu().numpy().T])

    shortest_path = []
    for x in neg_edge_index.cpu().numpy().T:
        try:
            shortest_path.append(len(nx.shortest_path(G, x[0], x[1])))
        except:
            shortest_path.append(999)

    neg = neg[np.array(shortest_path) < 4]

    neg_edge_index = neg_edge_index[:,np.array(shortest_path) < 4]

    print(len(neg))
    print(len(pos))

    all_cn = [pos, neg]
    max_cn = max(map(lambda x: max(x), all_cn))
    min_cn = min(map(lambda x: min(x), all_cn))
    print(len(list(nx.isolates(G))))

    labels = ['pos', 'neg']
    plt.figure(figsize=(8,6))
    plt.hist(all_cn, bins=100, histtype='bar', label=labels)
    plt.legend(loc="upper right")
    plt.title('Neg/Pos with combined common neighbors (SkipGNN paper)')
    plt.xlabel('Combined number of neighbors of A and B')
    plt.ylabel('Frequency')
    plt.show()

    return neg_edge_index
