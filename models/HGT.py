import torch.nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, MLP, Linear


class HGT(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            metadata: tuple,
            num_heads: int = 4,
            num_layers: int = 4,
    ):
        super().__init__()

        self.in_lins = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.in_lins[node_type] = MLP(
                in_channels=-1,
                hidden_channels=2 * hidden_channels,
                out_channels=hidden_channels,
                num_layers=3,
                norm="layer_norm",
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(
                hidden_channels, hidden_channels,
                metadata, num_heads, group='sum',
            ))

        self.out_lins = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.out_lins[node_type] = Linear(
                in_channels=hidden_channels,
                out_channels=out_channels,
            )

    def forward(self, data: HeteroData, **kwargs):
        x_dict = {t: data[t].x for t in data.node_types}
        edge_index_dict = {t: data[t].edge_index for t in data.edge_types}
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.in_lins[node_type](x)

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.out_lins[node_type](x)
        return x_dict
