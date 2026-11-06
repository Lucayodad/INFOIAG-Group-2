"""Export structural sources for report Figures 1-3; optionally render DOT to SVG.

    python export_report_figures.py --out results/report/figures [--render]

Figure 1 is exported from the compiled LangGraph. Figures 2 and 3 are read
directly from the unchanged OWL ontology. Layout can differ from the report's
artwork. Rendering Figures 2-3 requires the standard Graphviz `dot` executable.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from rdflib import OWL, RDF, RDFS, Graph, URIRef

from ollama_backend import OllamaClient
from ontology_tools import DEFAULT_ONTOLOGY, NS
from summary_agent import build_summary_agent


def export_figures(out, render=False):
    out.mkdir(parents=True, exist_ok=True)
    workflow = build_summary_agent(Graph(), OllamaClient(), proportion_check=True)
    drawing = workflow.get_graph()
    (out / "figure1_agent.mmd").write_text(drawing.draw_mermaid(), encoding="utf-8")
    lines = ['digraph Agent {', 'rankdir=TB;', 'node [shape=box, fontname="Arial"];']
    for name in sorted(drawing.nodes):
        shape = 'diamond' if name == 'route_checks' else 'ellipse' if name in ('__start__', '__end__') else 'box'
        lines.append(f'{json.dumps(name)} [shape={shape}];')
    for edge in drawing.edges:
        attributes = 'style=dashed' if edge.conditional else 'style=solid'
        lines.append(f'{json.dumps(edge.source)} -> {json.dumps(edge.target)} [{attributes}];')
    (out / 'figure1_agent.dot').write_text('\n'.join(lines + ['}']), encoding='utf-8')
    graph = Graph().parse(DEFAULT_ONTOLOGY, format="xml")
    classes = sorted({c for c in graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef) and str(c).startswith(str(NS))})
    label = lambda node: str(node).rsplit('#', 1)[-1]
    quote = lambda node: json.dumps(label(node))
    lines = ['digraph ClassHierarchy {', 'rankdir=BT;', 'node [shape=box, style="rounded,filled", fontname="Arial"];']
    for node in classes:
        color = '#ffeb9c' if label(node) in ('Person', 'Company', 'Country', 'Sport', 'Health') else '#dceeff'
        lines.append(f'{quote(node)} [fillcolor="{color}"];')
    for child, parent in sorted(graph.subject_objects(RDFS.subClassOf)):
        if child in classes and parent in classes:
            lines.append(f'{quote(child)} -> {quote(parent)};')
    (out / 'figure2_classes.dot').write_text('\n'.join(lines + ['}']), encoding='utf-8')
    lines = ['digraph ObjectProperties {', 'rankdir=LR;', 'node [shape=box, fontname="Arial"];', 'edge [fontname="Arial", fontsize=9];']
    characteristics = ((OWL.SymmetricProperty, 'sym.'), (OWL.TransitiveProperty, 'trans.'),
                       (OWL.FunctionalProperty, 'func.'), (OWL.InverseFunctionalProperty, 'inv.func.'))
    for prop in sorted(set(graph.subjects(RDF.type, OWL.ObjectProperty))):
        note = ', '.join(short for kind, short in characteristics if (prop, RDF.type, kind) in graph)
        text = label(prop) + (f' ({note})' if note else '')
        for domain in graph.objects(prop, RDFS.domain):
            for target in graph.objects(prop, RDFS.range):
                if domain in classes and target in classes:
                    lines.append(f'{quote(domain)} -> {quote(target)} [label={json.dumps(text)}];')
    (out / 'figure3_properties.dot').write_text('\n'.join(lines + ['}']), encoding='utf-8')
    if render:
        dot = shutil.which('dot')
        if not dot:
            raise RuntimeError('Install Graphviz and add dot to PATH to render the supplied DOT sources.')
        for path in sorted(out.glob('*.dot')):
            subprocess.run([dot, '-Tsvg', str(path), '-o', str(path.with_suffix('.svg'))], check=True)
    print(f'Exported Figures 1-3 from the code and ontology to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/report/figures'))
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    export_figures(args.out, args.render)
