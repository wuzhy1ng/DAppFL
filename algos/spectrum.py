import torch.nn
from torch_geometric.data import HeteroData


class ProgramSpectrum(torch.nn.Module):
    def forward(self, data: HeteroData):
        spectrum = torch.zeros(size=(
            data.num_nodes, data['Case'].num_nodes
        ))

        code_types = data.node_types
        code_types.remove('Case')
        start_idx = 0
        for code_type in code_types:
            case_edge_type = sorted(['Case', code_type])
            edge_type = (case_edge_type[0], 'to', case_edge_type[1])
            if len(data[edge_type]) == 0:
                continue
            case_row_idx = case_edge_type.index('Case')
            code_row_idx = case_edge_type.index(code_type)
            edge_index = data[edge_type].edge_index
            for i in range(edge_index.size()[1]):
                case_idx = edge_index[case_row_idx][i]
                code_idx = edge_index[code_row_idx][i] + start_idx
                if data['Case'].y[case_idx] == 1 and data[code_type].y[edge_index[code_row_idx][i]] == 1:
                    spectrum[code_idx, case_idx] = -1
                else:
                    spectrum[code_idx, case_idx] += 1
            start_idx += data[code_type].num_nodes

        return spectrum
