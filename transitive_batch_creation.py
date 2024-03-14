import math

UNKNOWN_VALUE = 0

def get_batch_sims(queries, galleries, labels_dict, G, transitive_weight_beta):
    """Main method to calculating the TCL batch labels

    Args:
        queries (List[Path]): A list of paths of each query image in the batch.
        galleries (List[Path]): A list of paths of each gallery image in the batch.
        labels_dict (dict): Dictionary of GT labels in the form of (query, gallery): label (0 or 1)
        G (_type_): _description_
        transitive_weight_beta (_type_): Controls exponential decay of the transitive weight.

    Returns:
        _type_: _description_
    """
    query_labels = [
        get_batch_item_sims(query, queries, galleries, labels_dict, G, transitive_weight_beta) for query in queries
    ]
    gallery_labels = [
        get_batch_item_sims(gallery, queries, galleries, labels_dict, G, transitive_weight_beta)
        for gallery in galleries
    ]

    all_labels = [*query_labels, *gallery_labels]

    return all_labels


def get_batch_item_sims(item, queries, galleries, labels_dict, G, transitive_weight_beta):
    query_values = [
        compare_batch_item_sims(item, query, labels_dict, G, beta=transitive_weight_beta) for query in queries
    ]
    gallery_values = [
        compare_batch_item_sims(item, gallery, labels_dict, G, beta=transitive_weight_beta) for gallery in galleries
    ]

    values = [*query_values, *gallery_values]

    return values


def compare_batch_item_sims(item, other_item, labels_dict, G, beta=0.7):
    value = UNKNOWN_VALUE

    if item == other_item:
        value = 1
    else:
        try:
            key = tuple({item, other_item})
            
            if key in labels_dict:
                value = labels_dict.get(key)
            else:
                path_length = get_shortest_length(G, item, other_item)
                
                if path_length == math.inf:
                    value == 0
                else:
                    if beta == 0:
                        value = 1 # When beta=1, we treat transitive labels as "non-soft" - the same as annotated positives, Note that this is only relevant if max-positive-length > 1
                    else:
                        value = math.exp(-beta * path_length)
                    
        except KeyError:
            pass

    return value

def get_shortest_length(G, node1, node2):
    result = None
    
    if G.has_edge(node1, node2):
        result = G[node1][node2]['weight']
    else:
        return math.inf
        
    return result
