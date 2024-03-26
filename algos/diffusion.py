from typing import Dict, Callable, Any

import networkx as nx


def leak_diffusion(
        graph: nx.DiGraph, sources: set,
        gamma: float = 0.1, epsilon: float = 1e-3,
        grad_func: Callable = lambda _x: _x ** 3,
) -> Dict:
    """
    Non-Linear graph diffusion for the leakage money tracing.

    :param graph: the money graph
    :param sources: the leaky records
    :param gamma: the hyper-parameter, in (0, 1)
    :param epsilon: the hyper-parameter, in (0, 1)
    :param grad_func: the gradient function, can be p-norm function
    :return: The importance of each money transfer record
    """
    x = dict()
    r = {s: 1.0 for s in sources}

    q = [s for s in sources]
    while len(q) > 0:
        u = q.pop(0)
        if r[u] <= epsilon:
            continue

        # bi-search and updating for the residual
        dx_max = 1 - x.get(u, 0)
        dx = dx_max / 2
        is_source = (u in sources)
        ru_new = _eval_residual(
            graph=graph, u=u, x=x, dx=dx, is_source=is_source,
            gamma=gamma, grad_func=grad_func,
        )
        while ru_new > 1 - epsilon:
            dx = (dx + dx_max) / 2
            ru_new = _eval_residual(
                graph=graph, u=u, x=x, dx=dx, is_source=is_source,
                gamma=gamma, grad_func=grad_func,
            )
        r[u] = ru_new

        # push to input neighbors
        dist = {
            v: attr.get('weight', 0)
            for v, _, attr in graph.in_edges(u, data=True)
        }
        sum_weight = sum(dist.values())
        for v, w in dist.items():
            delta = grad_func(x.get(v, 0) - x.get(u, 0)) - grad_func(x.get(v, 0) - (x.get(u, 0) + dx))
            r[v] = r.get(v, 0) + (w / sum_weight) * delta / gamma
            if r[v] > epsilon:
                q.append(v)

        # push to self
        x[u] = x.get(u, 0) + dx

    # return the normalized result
    sum_x = sum(x.values())
    return {i: xi / sum_x for i, xi in x.items()}


def _eval_residual(
        graph: nx.DiGraph, u: Any, x: Dict, dx: float, is_source: bool,
        gamma: float, grad_func: Callable,
):
    if is_source:
        return grad_func(x.get(u, 0) + dx - 1)

    rlt = 0
    dist = {v: attr.get('weight', 0) for _, v, attr in graph.out_edges(u, data=True)}
    sum_weight = sum(dist.values())
    for v, w in dist.items():
        rlt += grad_func(x.get(u, 0) + dx - x.get(v, 0)) * (w / sum_weight)
    return -rlt / gamma


if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_edges_from([
        ('A', 'C', {'weight': 0.3}),
        ('A', 'D', {'weight': 0.3}),
        ('A', 'E', {'weight': 0.3}),
        ('C', 'B', {'weight': 1.0}),
        ('D', 'F', {'weight': 1.0}),
        ('F', 'B', {'weight': 1.0}),
        ('E', 'G', {'weight': 1.0}),
        ('G', 'F', {'weight': 0.2}),
        ('G', 'H', {'weight': 0.8}),
        ('H', 'B', {'weight': 1.0}),
    ])
    sources = {'B'}
    x = leak_diffusion(g, sources)
    print(x)

    top_k = 5
    x_rank = sorted(list(x.items()), key=lambda item: item[1], reverse=True)
    sub_g = g.subgraph([node for node, _ in x_rank[:top_k]])
    for u, v, attr in sub_g.edges(data=True):
        print('{}->{}, {}'.format(u, v, attr))
