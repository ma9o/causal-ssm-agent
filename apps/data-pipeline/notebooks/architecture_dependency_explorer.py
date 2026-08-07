import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import ast
    import importlib.util
    import math
    from collections import defaultdict, deque
    from dataclasses import dataclass
    from pathlib import Path

    import networkx as nx
    import numpy as np
    import plotly.graph_objects as go

    return (
        Path,
        ast,
        dataclass,
        defaultdict,
        deque,
        go,
        importlib,
        math,
        np,
        nx,
    )


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Architecture dependency explorer

    This notebook turns the Python source tree into an **executable architecture map**. It
    parses imports without importing application modules, so exploring the graph does not
    initialize JAX, NumPyro, Temporal, Prefect, or any service clients.

    Use the two lenses together:

    - **Current package layout** shows the physical organization that exists today.
    - **Proposed responsibility boundaries** classifies the same files into the candidate
      domain, identification, structural front, SSM model, exact inference, analysis, and
      application layers.

    Every arrow points from the **importer to the module it imports**. In the proposed view,
    red arrows violate the candidate dependency policy. The context-footprint section treats
    another cluster as an opaque public boundary, giving an estimate of how much implementation
    context a hard package seam could remove.
    """)
    return


@app.cell
def source_location(Path):
    source_root = Path(__file__).resolve().parents[1] / "src" / "nof1_causal_lab"
    return (source_root,)


@app.cell
def dependency_model(ast, dataclass, importlib):
    @dataclass(frozen=True, slots=True)
    class ModuleInfo:
        name: str
        path: str
        relative_path: str
        loc: int
        is_package: bool

    @dataclass(frozen=True, slots=True)
    class ImportRef:
        importer: str
        imported: str
        kind: str
        lineno: int
        symbol: str

    def _module_name(source_root, path):
        relative = path.relative_to(source_root)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        suffix = ".".join(parts)
        name = "nof1_causal_lab" if not suffix else f"nof1_causal_lab.{suffix}"
        return name, is_package

    def _nearest_internal_module(target, module_names):
        candidate = target
        while candidate.startswith("nof1_causal_lab"):
            if candidate in module_names:
                return candidate
            if "." not in candidate:
                break
            candidate = candidate.rpartition(".")[0]
        return None

    def _is_type_checking_test(node):
        if isinstance(node, ast.Name):
            return node.id == "TYPE_CHECKING"
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "typing"
            and node.attr == "TYPE_CHECKING"
        )

    class _ImportVisitor(ast.NodeVisitor):
        def __init__(self, *, importer, is_package, module_names):
            self.importer = importer
            self.package = importer if is_package else importer.rpartition(".")[0]
            self.module_names = module_names
            self.type_checking_depth = 0
            self.refs = []

        @property
        def kind(self):
            return "type_checking" if self.type_checking_depth else "runtime"

        def _append(self, *, target, lineno, symbol):
            resolved = _nearest_internal_module(target, self.module_names)
            if resolved is not None and resolved != self.importer:
                self.refs.append(
                    ImportRef(
                        importer=self.importer,
                        imported=resolved,
                        kind=self.kind,
                        lineno=lineno,
                        symbol=symbol,
                    )
                )

        def visit_If(self, node):
            if not _is_type_checking_test(node.test):
                self.generic_visit(node)
                return

            self.visit(node.test)
            self.type_checking_depth += 1
            for statement in node.body:
                self.visit(statement)
            self.type_checking_depth -= 1
            for statement in node.orelse:
                self.visit(statement)

        def visit_Import(self, node):
            for alias in node.names:
                self._append(target=alias.name, lineno=node.lineno, symbol=alias.name)

        def visit_ImportFrom(self, node):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                base = importlib.util.resolve_name(relative_name, self.package)
            else:
                base = node.module or ""

            for alias in node.names:
                submodule = f"{base}.{alias.name}"
                target = submodule if submodule in self.module_names else base
                self._append(target=target, lineno=node.lineno, symbol=alias.name)

    def build_inventory(source_root):
        paths = sorted(source_root.rglob("*.py"))
        module_rows = []
        for path in paths:
            name, is_package = _module_name(source_root, path)
            text = path.read_text(encoding="utf-8")
            module_rows.append(
                ModuleInfo(
                    name=name,
                    path=str(path),
                    relative_path=path.relative_to(source_root).as_posix(),
                    loc=len(text.splitlines()),
                    is_package=is_package,
                )
            )

        module_names = frozenset(row.name for row in module_rows)
        refs = []
        for row in module_rows:
            text = source_root.joinpath(row.relative_path).read_text(encoding="utf-8")
            tree = ast.parse(text, filename=row.path)
            visitor = _ImportVisitor(
                importer=row.name,
                is_package=row.is_package,
                module_names=module_names,
            )
            visitor.visit(tree)
            refs.extend(visitor.refs)
        return tuple(module_rows), tuple(refs)

    return ImportRef, ModuleInfo, build_inventory


@app.cell
def cluster_model():
    proposed_order = (
        "domain",
        "identification",
        "structural-front",
        "ssm-model",
        "ssm-inference",
        "analysis",
        "application",
    )
    proposed_colors = {
        "domain": "#4C78A8",
        "identification": "#72B7B2",
        "structural-front": "#54A24B",
        "ssm-model": "#F2CF5B",
        "ssm-inference": "#F58518",
        "analysis": "#B279A2",
        "application": "#9D755D",
    }
    current_colors = {
        "artifacts": "#4C78A8",
        "models.ssm": "#F2CF5B",
        "models.other": "#B279A2",
        "flows": "#E45756",
        "machine": "#9D755D",
        "workers": "#FF9DA6",
        "utils": "#72B7B2",
        "package root": "#BAB0AC",
    }
    allowed_proposed_dependencies = {
        "domain": frozenset({"domain"}),
        "identification": frozenset({"domain", "identification"}),
        "structural-front": frozenset({"domain", "identification", "structural-front"}),
        "ssm-model": frozenset({"domain", "ssm-model"}),
        "ssm-inference": frozenset({"domain", "ssm-model", "ssm-inference"}),
        "analysis": frozenset(
            {"domain", "identification", "ssm-model", "ssm-inference", "analysis"}
        ),
        "application": frozenset(proposed_order),
    }

    def current_cluster(module):
        prefix = "nof1_causal_lab."
        if not module.startswith(prefix):
            return "package root"
        relative = module.removeprefix(prefix)
        first = relative.split(".", maxsplit=1)[0]
        if relative == "models" or relative.startswith("models."):
            return "models.ssm" if relative.startswith("models.ssm") else "models.other"
        if first in {"artifacts", "flows", "machine", "workers", "utils"}:
            return first
        return "package root"

    def proposed_cluster(module):
        domain_prefixes = (
            "nof1_causal_lab.artifacts",
            "nof1_causal_lab.compilation_errors",
            "nof1_causal_lab.distributions",
            "nof1_causal_lab.json_types",
            "nof1_causal_lab.measurement_types",
            "nof1_causal_lab.utils.observation_semantics",
            "nof1_causal_lab.utils.structural_plan",
        )
        identification_prefixes = (
            "nof1_causal_lab.utils.causal_design",
            "nof1_causal_lab.utils.estimation_projection",
            "nof1_causal_lab.utils.identifiability",
        )
        structural_front_prefixes = ("nof1_causal_lab.models.structural",)
        analysis_prefixes = (
            "nof1_causal_lab.models.causal_proofs",
            "nof1_causal_lab.models.posterior_predictive",
            "nof1_causal_lab.models.predictive_simulation",
            "nof1_causal_lab.models.ssm.construct_admission",
            "nof1_causal_lab.models.ssm.counterfactual",
            "nof1_causal_lab.models.ssm.predictive",
        )
        ssm_model_prefixes = (
            "nof1_causal_lab.models.model_semantics",
            "nof1_causal_lab.models.ssm",
        )

        if module.startswith(domain_prefixes):
            return "domain"
        if module.startswith(identification_prefixes):
            return "identification"
        if module.startswith(structural_front_prefixes):
            return "structural-front"
        if module.startswith("nof1_causal_lab.models.ssm.inference") or module.startswith(
            "nof1_causal_lab.sampler_config"
        ):
            return "ssm-inference"
        if module.startswith(analysis_prefixes):
            return "analysis"
        if module.startswith(ssm_model_prefixes):
            return "ssm-model"
        return "application"

    return (
        allowed_proposed_dependencies,
        current_cluster,
        current_colors,
        proposed_cluster,
        proposed_colors,
        proposed_order,
    )


@app.cell
def graph_analysis(defaultdict, deque, math, nx):
    def build_module_graph(module_infos, import_refs):
        graph = nx.DiGraph()
        graph.add_nodes_from(row.name for row in module_infos)
        for ref in import_refs:
            if graph.has_edge(ref.importer, ref.imported):
                graph[ref.importer][ref.imported]["references"] += 1
            else:
                graph.add_edge(ref.importer, ref.imported, references=1)
        return graph

    def aggregate_clusters(module_infos, import_refs, clusterer, allowed_dependencies):
        stats = defaultdict(lambda: {"modules": 0, "loc": 0})
        for row in module_infos:
            cluster = clusterer(row.name)
            stats[cluster]["modules"] += 1
            stats[cluster]["loc"] += row.loc

        pairs = defaultdict(list)
        for ref in import_refs:
            importer_cluster = clusterer(ref.importer)
            imported_cluster = clusterer(ref.imported)
            pairs[(importer_cluster, imported_cluster)].append(ref)

        forbidden_pairs = set()
        if allowed_dependencies is not None:
            runtime_pairs = {
                (clusterer(ref.importer), clusterer(ref.imported))
                for ref in import_refs
                if ref.kind == "runtime"
            }
            for importer_cluster, imported_cluster in runtime_pairs:
                if (
                    importer_cluster != imported_cluster
                    and imported_cluster not in allowed_dependencies[importer_cluster]
                ):
                    forbidden_pairs.add((importer_cluster, imported_cluster))
        return dict(stats), dict(pairs), forbidden_pairs

    def boundary_rows(import_refs, clusterer, forbidden_pairs):
        grouped = defaultdict(list)
        for ref in import_refs:
            importer_cluster = clusterer(ref.importer)
            imported_cluster = clusterer(ref.imported)
            if importer_cluster != imported_cluster:
                grouped[(ref.importer, ref.imported)].append(ref)

        rows = []
        for (importer, imported), refs in grouped.items():
            importer_cluster = clusterer(importer)
            imported_cluster = clusterer(imported)
            pair = (importer_cluster, imported_cluster)
            rows.append(
                {
                    "status": "forbidden" if pair in forbidden_pairs else "allowed",
                    "importer cluster": importer_cluster,
                    "importer": importer.removeprefix("nof1_causal_lab."),
                    "imported cluster": imported_cluster,
                    "imported": imported.removeprefix("nof1_causal_lab."),
                    "references": len(refs),
                    "runtime": sum(ref.kind == "runtime" for ref in refs),
                    "type-only": sum(ref.kind == "type_checking" for ref in refs),
                    "lines": ", ".join(str(line) for line in sorted({ref.lineno for ref in refs})),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["status"] != "forbidden",
                -row["references"],
                row["importer"],
                row["imported"],
            ),
        )

    def cycle_rows(module_graph, module_lookup, clusterer):
        components = [
            component
            for component in nx.strongly_connected_components(module_graph)
            if len(component) > 1
        ]
        rows = []
        for component in components:
            ordered = sorted(component)
            rows.append(
                {
                    "modules": len(ordered),
                    "LOC": sum(module_lookup[name].loc for name in ordered),
                    "clusters": ", ".join(sorted({clusterer(name) for name in ordered})),
                    "members": ", ".join(name.removeprefix("nof1_causal_lab.") for name in ordered),
                }
            )
        return sorted(rows, key=lambda row: (-row["modules"], -row["LOC"], row["members"]))

    def dependency_footprint(module_graph, root, clusterer):
        full = {root, *nx.descendants(module_graph, root)}
        root_cluster = clusterer(root)
        local = {root}
        boundary = set()
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for dependency in module_graph.successors(current):
                if clusterer(dependency) == root_cluster:
                    if dependency not in local:
                        local.add(dependency)
                        queue.append(dependency)
                else:
                    boundary.add(dependency)
        bounded = local | boundary
        return full, bounded, local, boundary

    def cluster_sort(clusters, preferred_order):
        preferred_index = {name: index for index, name in enumerate(preferred_order)}
        return sorted(clusters, key=lambda name: (preferred_index.get(name, math.inf), name))

    def footprint_rows(nodes, module_lookup, clusterer):
        grouped = defaultdict(lambda: {"modules": 0, "loc": 0})
        for name in nodes:
            cluster = clusterer(name)
            grouped[cluster]["modules"] += 1
            grouped[cluster]["loc"] += module_lookup[name].loc
        return dict(grouped)

    return (
        aggregate_clusters,
        boundary_rows,
        build_module_graph,
        cluster_sort,
        cycle_rows,
        dependency_footprint,
        footprint_rows,
    )


@app.cell
def figure_builders(go, math, np, nx):
    def make_module_treemap(module_infos, clusterer, colors):
        cluster_names = sorted({clusterer(row.name) for row in module_infos})
        ids = [f"cluster::{cluster}" for cluster in cluster_names]
        labels = list(cluster_names)
        parents = [""] * len(cluster_names)
        values = [
            sum(row.loc for row in module_infos if clusterer(row.name) == cluster)
            for cluster in cluster_names
        ]
        hover = [
            (
                f"{sum(clusterer(row.name) == cluster for row in module_infos):,} modules"
                f"<br>{value:,} LOC"
            )
            for cluster, value in zip(cluster_names, values, strict=True)
        ]
        marker_colors = [colors[cluster] for cluster in cluster_names]

        for row in sorted(module_infos, key=lambda item: (clusterer(item.name), item.name)):
            cluster = clusterer(row.name)
            ids.append(row.name)
            labels.append(row.name.removeprefix("nof1_causal_lab."))
            parents.append(f"cluster::{cluster}")
            values.append(row.loc)
            hover.append(f"{row.relative_path}<br>{row.loc:,} LOC")
            marker_colors.append(colors[cluster])

        figure = go.Figure(
            go.Treemap(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker={"colors": marker_colors, "line": {"color": "white", "width": 1}},
                customdata=hover,
                hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
                textinfo="label+value",
                pathbar={"visible": True},
            )
        )
        figure.update_layout(
            title="Source organization · area is lines of code",
            height=680,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
        )
        return figure

    def make_cluster_graph(cluster_stats, cluster_pairs, forbidden_pairs, colors):
        graph = nx.DiGraph()
        graph.add_nodes_from(cluster_stats)
        for pair, refs in cluster_pairs.items():
            if pair[0] != pair[1]:
                graph.add_edge(*pair, weight=len(refs))

        positions = nx.spring_layout(
            graph.to_undirected(),
            seed=17,
            k=1.8 / math.sqrt(max(len(graph), 1)),
            iterations=200,
            weight="weight",
        )
        figure = go.Figure()
        edge_x = []
        edge_y = []
        edge_text = []
        for importer, imported, data in graph.edges(data=True):
            x0, y0 = positions[importer]
            x1, y1 = positions[imported]
            count = data["weight"]
            forbidden = (importer, imported) in forbidden_pairs
            color = "#D62728" if forbidden else "#7A7A7A"
            width = min(7.0, 1.0 + math.log1p(count))
            figure.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.1,
                arrowwidth=width,
                arrowcolor=color,
                opacity=0.8,
                standoff=23,
                startstandoff=23,
            )
            edge_x.append((x0 + x1) / 2)
            edge_y.append((y0 + y1) / 2)
            status = "forbidden" if forbidden else "allowed"
            edge_text.append(f"{importer} → {imported}<br>{count:,} import references<br>{status}")

        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="markers",
                marker={"size": 18, "color": "rgba(0,0,0,0.01)"},
                text=edge_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

        ordered_nodes = sorted(graph.nodes)
        max_loc = max((cluster_stats[node]["loc"] for node in ordered_nodes), default=1)
        sizes = [
            34 + 46 * math.sqrt(cluster_stats[node]["loc"] / max_loc) for node in ordered_nodes
        ]
        figure.add_trace(
            go.Scatter(
                x=[positions[node][0] for node in ordered_nodes],
                y=[positions[node][1] for node in ordered_nodes],
                mode="markers+text",
                text=ordered_nodes,
                textposition="bottom center",
                marker={
                    "size": sizes,
                    "color": [colors[node] for node in ordered_nodes],
                    "line": {"color": "white", "width": 2},
                },
                customdata=[
                    [
                        cluster_stats[node]["modules"],
                        cluster_stats[node]["loc"],
                    ]
                    for node in ordered_nodes
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>%{customdata[0]:,} modules"
                    "<br>%{customdata[1]:,} LOC<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.update_layout(
            title="Cluster dependencies · arrow points from importer to imported dependency",
            height=650,
            margin={"l": 20, "r": 20, "t": 55, "b": 25},
            xaxis={"visible": False},
            yaxis={"visible": False},
            plot_bgcolor="white",
            hovermode="closest",
        )
        return figure

    def make_dependency_matrix(
        clusters,
        cluster_pairs,
        forbidden_pairs,
    ):
        index = {cluster: position for position, cluster in enumerate(clusters)}
        raw = np.zeros((len(clusters), len(clusters)), dtype=int)
        for (importer, imported), refs in cluster_pairs.items():
            raw[index[importer], index[imported]] = len(refs)

        figure = go.Figure(
            go.Heatmap(
                z=np.log1p(raw),
                x=list(range(len(clusters))),
                y=list(range(len(clusters))),
                customdata=raw,
                text=raw,
                texttemplate="%{text}",
                colorscale=[
                    [0.0, "#F7FBFF"],
                    [0.25, "#C6DBEF"],
                    [0.5, "#6BAED6"],
                    [0.75, "#3182BD"],
                    [1.0, "#08519C"],
                ],
                colorbar={"title": "log(1 + refs)"},
                hovertemplate=(
                    "importer: %{y}<br>imported: %{x}<br>%{customdata:,} references<extra></extra>"
                ),
            )
        )
        for importer, imported in forbidden_pairs:
            row = index[importer]
            column = index[imported]
            if raw[row, column]:
                figure.add_shape(
                    type="rect",
                    x0=column - 0.48,
                    x1=column + 0.48,
                    y0=row - 0.48,
                    y1=row + 0.48,
                    line={"color": "#D62728", "width": 4},
                    fillcolor="rgba(0,0,0,0)",
                )
        figure.update_layout(
            title="Import-reference matrix · rows import columns · red outline is forbidden",
            height=max(520, 62 * len(clusters)),
            margin={"l": 150, "r": 30, "t": 60, "b": 130},
            xaxis={
                "title": "imported dependency",
                "tickmode": "array",
                "tickvals": list(range(len(clusters))),
                "ticktext": clusters,
                "tickangle": -35,
            },
            yaxis={
                "title": "importer",
                "tickmode": "array",
                "tickvals": list(range(len(clusters))),
                "ticktext": clusters,
                "autorange": "reversed",
            },
        )
        return figure

    def make_footprint_treemap(nodes, module_lookup, clusterer, colors, title):
        cluster_names = sorted({clusterer(name) for name in nodes})
        ids = [f"footprint::{cluster}" for cluster in cluster_names]
        labels = list(cluster_names)
        parents = [""] * len(cluster_names)
        values = [
            sum(module_lookup[name].loc for name in nodes if clusterer(name) == cluster)
            for cluster in cluster_names
        ]
        marker_colors = [colors[cluster] for cluster in cluster_names]
        for name in sorted(nodes, key=lambda item: (clusterer(item), item)):
            cluster = clusterer(name)
            ids.append(name)
            labels.append(name.removeprefix("nof1_causal_lab."))
            parents.append(f"footprint::{cluster}")
            values.append(module_lookup[name].loc)
            marker_colors.append(colors[cluster])

        figure = go.Figure(
            go.Treemap(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker={"colors": marker_colors, "line": {"color": "white", "width": 1}},
                hovertemplate="<b>%{label}</b><br>%{value:,} LOC<extra></extra>",
                textinfo="label+value",
            )
        )
        figure.update_layout(
            title=title,
            height=620,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
        )
        return figure

    def make_footprint_comparison(full, bounded, module_lookup):
        full_loc = sum(module_lookup[name].loc for name in full)
        bounded_loc = sum(module_lookup[name].loc for name in bounded)
        figure = go.Figure(
            go.Bar(
                x=["full transitive closure", "boundary-scoped estimate"],
                y=[full_loc, bounded_loc],
                text=[
                    f"{len(full):,} modules<br>{full_loc:,} LOC",
                    f"{len(bounded):,} modules<br>{bounded_loc:,} LOC",
                ],
                textposition="auto",
                marker={"color": ["#7A7A7A", "#4C78A8"]},
                hovertemplate="%{x}<br>%{y:,} LOC<extra></extra>",
            )
        )
        figure.update_layout(
            title="Estimated reasoning context",
            height=420,
            margin={"l": 60, "r": 20, "t": 55, "b": 70},
            yaxis={"title": "lines of code"},
            showlegend=False,
        )
        return figure

    return (
        make_cluster_graph,
        make_dependency_matrix,
        make_footprint_comparison,
        make_footprint_treemap,
        make_module_treemap,
    )


@app.cell
def source_inventory(build_inventory, source_root):
    module_infos, import_refs = build_inventory(source_root)
    module_lookup = {row.name: row for row in module_infos}
    return import_refs, module_infos, module_lookup


@app.cell(hide_code=True)
def controls(import_refs, mo):
    scheme_control = mo.ui.radio(
        options={
            "Current package layout": "current",
            "Proposed responsibility boundaries": "proposed",
        },
        value="Proposed responsibility boundaries",
        inline=True,
        label="Organization lens",
    )
    import_scope_control = mo.ui.radio(
        options={
            "Runtime imports": "runtime",
            "Runtime + TYPE_CHECKING": "all",
        },
        value="Runtime imports",
        inline=True,
        label="Import scope",
    )
    _runtime_count = sum(ref.kind == "runtime" for ref in import_refs)
    _type_count = sum(ref.kind == "type_checking" for ref in import_refs)
    mo.vstack(
        [
            mo.hstack(
                [scheme_control, import_scope_control],
                justify="start",
                gap=3,
            ),
            mo.md(
                f"**{_runtime_count:,}** runtime import references · "
                f"**{_type_count:,}** type-checking references"
            ),
        ]
    )
    return import_scope_control, scheme_control


@app.cell
def active_architecture(
    aggregate_clusters,
    allowed_proposed_dependencies,
    build_module_graph,
    cluster_sort,
    current_cluster,
    current_colors,
    import_refs,
    import_scope_control,
    module_infos,
    proposed_cluster,
    proposed_colors,
    proposed_order,
    scheme_control,
):
    active_import_refs = tuple(
        ref for ref in import_refs if import_scope_control.value == "all" or ref.kind == "runtime"
    )
    if scheme_control.value == "proposed":
        clusterer = proposed_cluster
        cluster_colors = proposed_colors
        allowed_dependencies = allowed_proposed_dependencies
        preferred_order = proposed_order
        scheme_label = "Proposed responsibility boundaries"
    else:
        clusterer = current_cluster
        cluster_colors = current_colors
        allowed_dependencies = None
        preferred_order = tuple(current_colors)
        scheme_label = "Current package layout"

    module_graph = build_module_graph(module_infos, active_import_refs)
    cluster_stats, cluster_pairs, forbidden_pairs = aggregate_clusters(
        module_infos,
        active_import_refs,
        clusterer,
        allowed_dependencies,
    )
    clusters = cluster_sort(cluster_stats, preferred_order)
    return (
        active_import_refs,
        cluster_colors,
        cluster_pairs,
        cluster_stats,
        clusterer,
        clusters,
        forbidden_pairs,
        module_graph,
        scheme_label,
    )


@app.cell(hide_code=True)
def overview_metrics(
    active_import_refs,
    clusterer,
    forbidden_pairs,
    mo,
    module_graph,
    module_infos,
    nx,
    scheme_label,
):
    _cross_boundary = sum(
        clusterer(ref.importer) != clusterer(ref.imported) for ref in active_import_refs
    )
    _forbidden = sum(
        (clusterer(ref.importer), clusterer(ref.imported)) in forbidden_pairs
        for ref in active_import_refs
    )
    _cycles = [
        component
        for component in nx.strongly_connected_components(module_graph)
        if len(component) > 1
    ]
    _policy_note = (
        "Current layout is descriptive, so no edges are marked forbidden."
        if not forbidden_pairs and scheme_label == "Current package layout"
        else "Forbidden references are evaluated against the candidate dependency policy."
    )
    mo.md(
        f"""
        ## Overview · {scheme_label}

        | modules | Python LOC | import references | distinct module edges | cross-cluster refs | forbidden refs | cyclic components |
        |---:|---:|---:|---:|---:|---:|---:|
        | {len(module_infos):,} | {sum(row.loc for row in module_infos):,} | {len(active_import_refs):,} | {module_graph.number_of_edges():,} | {_cross_boundary:,} | {_forbidden:,} | {len(_cycles):,} |

        {_policy_note}
        """
    )
    return


@app.cell(hide_code=True)
def overview_views(
    active_import_refs,
    boundary_rows,
    cluster_colors,
    cluster_pairs,
    cluster_stats,
    clusterer,
    clusters,
    cycle_rows,
    forbidden_pairs,
    make_cluster_graph,
    make_dependency_matrix,
    make_module_treemap,
    mo,
    module_graph,
    module_infos,
    module_lookup,
):
    _module_map = make_module_treemap(module_infos, clusterer, cluster_colors)
    _cluster_graph = make_cluster_graph(
        cluster_stats,
        cluster_pairs,
        forbidden_pairs,
        cluster_colors,
    )
    _matrix = make_dependency_matrix(clusters, cluster_pairs, forbidden_pairs)
    _boundary_data = boundary_rows(active_import_refs, clusterer, forbidden_pairs)
    _cycle_data = cycle_rows(module_graph, module_lookup, clusterer)
    _boundary_table = mo.ui.table(
        _boundary_data,
        selection=None,
        page_size=18,
        show_column_summaries=False,
        show_data_types=False,
        wrapped_columns=["importer", "imported"],
    )
    _cycle_table = (
        mo.ui.table(
            _cycle_data,
            selection=None,
            page_size=12,
            show_column_summaries=False,
            show_data_types=False,
            wrapped_columns=["members"],
        )
        if _cycle_data
        else mo.md("No strongly connected module components under this import scope.")
    )
    mo.ui.tabs(
        {
            "Module map": _module_map,
            "Cluster dependencies": _cluster_graph,
            "Dependency matrix": _matrix,
            "Boundary edges": _boundary_table,
            "Cycles": _cycle_table,
        }
    )
    return


@app.cell(hide_code=True)
def boundary_policy_md(mo):
    mo.md(r"""
    ### Candidate policy used by the proposed view

    - `domain` imports only itself.
    - `identification` imports `domain`.
    - `structural-front` imports `identification` and `domain`, then emits `StructuralPlan`.
    - `ssm-model` imports `domain`, but not identification or inference implementations.
    - `ssm-inference` imports `ssm-model` and `domain`.
    - `analysis` may join identification, model, inference, and domain evidence.
    - `application` may orchestrate every lower layer.

    The classifier is intentionally an architectural hypothesis over today's files. A mixed
    module remains classified as one unit, so a red edge can mean that the file itself needs to
    be split—for example, worker/UI prior metadata versus the compiler's authored-prior contract.
    """)
    return


@app.cell(hide_code=True)
def cluster_picker(clusters, mo, scheme_control):
    _preferred = "ssm-model" if scheme_control.value == "proposed" else "models.ssm"
    cluster_control = mo.ui.dropdown(
        options=clusters,
        value=_preferred,
        allow_select_none=False,
        searchable=False,
        label="Cluster",
    )
    mo.vstack(
        [
            mo.md("## Context footprint"),
            mo.md(
                "Choose an entry point. The full closure follows every internal dependency; "
                "the boundary-scoped estimate follows implementation inside the selected "
                "cluster but stops at the first imported module in another cluster."
            ),
            cluster_control,
        ]
    )
    return (cluster_control,)


@app.cell(hide_code=True)
def module_picker(cluster_control, clusterer, mo, module_infos):
    _names = sorted(
        row.name for row in module_infos if clusterer(row.name) == cluster_control.value
    )
    _compile_entry = "nof1_causal_lab.models.ssm.compile.artifact"
    _default = _compile_entry if _compile_entry in _names else _names[0]
    _options = {name.removeprefix("nof1_causal_lab."): name for name in _names}
    module_control = mo.ui.dropdown(
        options=_options,
        value=_default.removeprefix("nof1_causal_lab."),
        allow_select_none=False,
        searchable=True,
        full_width=True,
        label="Entry module",
    )
    mo.hstack([module_control], justify="start")
    return (module_control,)


@app.cell(hide_code=True)
def context_footprint(
    cluster_colors,
    clusterer,
    dependency_footprint,
    footprint_rows,
    make_footprint_comparison,
    make_footprint_treemap,
    mo,
    module_control,
    module_graph,
    module_lookup,
):
    _root = module_control.value
    _full, _bounded, _local, _boundary = dependency_footprint(
        module_graph,
        _root,
        clusterer,
    )
    _full_loc = sum(module_lookup[name].loc for name in _full)
    _bounded_loc = sum(module_lookup[name].loc for name in _bounded)
    _saved_loc = _full_loc - _bounded_loc
    _reduction = 0.0 if _full_loc == 0 else 100.0 * _saved_loc / _full_loc
    _direct_dependencies = module_graph.out_degree(_root)
    _direct_dependents = module_graph.in_degree(_root)
    _cluster_breakdown = footprint_rows(_full, module_lookup, clusterer)
    _breakdown_rows = [
        {
            "cluster": cluster,
            "modules": values["modules"],
            "LOC": values["loc"],
            "share": f"{100.0 * values['loc'] / _full_loc:.1f}%",
        }
        for cluster, values in sorted(
            _cluster_breakdown.items(),
            key=lambda item: -item[1]["loc"],
        )
    ]
    _comparison = make_footprint_comparison(_full, _bounded, module_lookup)
    _treemap = make_footprint_treemap(
        _full,
        module_lookup,
        clusterer,
        cluster_colors,
        "Full transitive dependency footprint · area is lines of code",
    )
    _breakdown_table = mo.ui.table(
        _breakdown_rows,
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ### `{_root.removeprefix("nof1_causal_lab.")}`

                | direct dependencies | direct dependents | full closure | boundary-scoped | estimated context reduction |
                |---:|---:|---:|---:|---:|
                | {_direct_dependencies:,} | {_direct_dependents:,} | {len(_full):,} modules / {_full_loc:,} LOC | {len(_bounded):,} modules / {_bounded_loc:,} LOC | {_reduction:.1f}% |

                The scoped estimate includes **{len(_local):,}** implementation modules from
                **{clusterer(_root)}** and **{len(_boundary):,}** first-contact modules at external
                boundaries. It does not claim those boundaries exist yet.
                """
            ),
            mo.ui.tabs(
                {
                    "Context comparison": _comparison,
                    "Dependency treemap": _treemap,
                    "Cluster breakdown": _breakdown_table,
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def interpretation_md(mo):
    mo.md(r"""
    ## How to use this in an architecture decision

    A promising package seam should improve several signals together:

    1. the dependency matrix is close to triangular;
    2. few high-weight red edges must be moved or inverted;
    3. strongly connected components do not cross the proposed boundary;
    4. boundary-scoped context is materially smaller than the full transitive closure; and
    5. the external first-contact modules form a small API that can be documented and tested.

    LOC and import counts are evidence about coupling, not a design verdict. A single import can
    carry a crucial invariant, while many imports may simply enumerate stable domain contracts.
    Use the boundary table to inspect the actual edges before turning a visual cluster into a
    physical package.
    """)
    return


if __name__ == "__main__":
    app.run()
