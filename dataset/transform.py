from torch_geometric.data import HeteroData


def format_data_type(data: HeteroData) -> HeteroData:
    for node_type in data.node_types:
        data[node_type].x = data[node_type].x.float()
        if node_type == 'Case':
            continue
        data[node_type].mask = data[node_type].x[:, 1] > 0

    for edge_type in data.edge_types:
        data[edge_type].edge_index = data[edge_type].edge_index.long()
        if data[edge_type].get('edge_attr') is None:
            continue
        data[edge_type].edge_attr = data[edge_type].edge_attr.float()

    return data
