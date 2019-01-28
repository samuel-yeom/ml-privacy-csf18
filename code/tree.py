from sklearn.tree import DecisionTreeRegressor, export_graphviz

def sklearn_train_tree(X, y, max_depth):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X, y)

def print_tree(tree, featnames):
    import pydotplus
    dot_data = export_graphviz(tree, out_file=None, feature_names=featnames)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('tree.pdf')
    print('Tree written to tree.pdf')