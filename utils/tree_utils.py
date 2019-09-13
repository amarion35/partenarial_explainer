import numpy as np
import copy
import json
import pandas as pd
import multiprocessing as mp

def list_paths(tree, node=None, parents=[]):
    if node is None:   
        node = tree[0]
    parents.append({
        'nodeid':node['nodeid'],
        'split':node['split'],
        'split_condition':node['split_condition'],
        'leaf': float(node['leaf'])
    })
    if node['is_leaf']:
        return [parents]
    yes_node = _get_node(tree, node['yes'])
    no_node = _get_node(tree, node['no'])
    yes_parents = copy.deepcopy(parents)
    yes_parents[-1]['condition'] = 1
    no_parents = copy.deepcopy(parents)
    no_parents[-1]['condition'] = -1
    yes_paths = list_paths(tree, yes_node, yes_parents)
    no_paths = list_paths(tree, no_node, no_parents)
    paths = []
    paths += yes_paths
    paths += no_paths
    return paths

def _get_node(tree, nodeid):
    if nodeid>=len(tree):
        node = next((node for node in tree if node['nodeid']==nodeid), None)
        return node
    node = tree[int(nodeid)]
    if node['nodeid']!=nodeid:
        node = next((node for node in tree if node['nodeid']==nodeid), None)
    return node

# Cast the model from a json format to a list of nodes
def model2table(booster):
    json_model = booster.get_dump(with_stats=True, dump_format='json')
    json_model = [json.loads(j_m) for j_m in json_model]
    
    def cast_tree_to_list(node):
        node_list = np.array([])
        node['is_leaf'] = True
        if 'children' in node:
            node['is_leaf'] = False
            for subnode in node['children']:
                subnode['parent'] = node['nodeid']
                node_list = np.append(node_list, cast_tree_to_list(subnode))
            del node['children']
            if isinstance(node['split'], str):
                node['split'] = int(node['split'].replace('f',''))
        else:
            node['split'] = np.nan
            node['split_condition'] = np.nan
            
        node_list = np.append(node_list, node)
        return node_list
    
    model = [list(cast_tree_to_list(j_m)) for j_m in json_model]
    model2 = []
    for m in model:
        df = pd.DataFrame(m)
        #df['split'] = df['split'].astype('int')
        df = df.sort_values('nodeid').to_dict('records')
        model2.append(df)
    return model2


def predict_model(model, X, n_class):
    pred = []
    p = mp.Pool(mp.cpu_count())
    pred = p.starmap(predict_tree, [(t, x) for x in X for t in model])
    p.close()
    pred = np.reshape(pred, (len(X), len(model)))
    res = [[np.sum(sample[i::n_class]) for i in range(n_class)] for sample in pred]
    return res

def predict_tree(tree, x):
    return float(browse_tree(tree, x, node=None, path=[])[-1]['leaf'])

def browse_tree(tree, x, node=None, path=[]):
    path = []
    if node is None:
        node = tree[0]
    while not node['is_leaf']:
        path.append(node)
        if x[int(node['split'])] < node['split_condition']:
            node = _get_node(tree, node['yes'])
        else:
            node = _get_node(tree, node['no'])
    path.append(node)
    return path


