# Knowledge Ripples project page

Static academic project page and browser-side FactProp popularity explorer.

## Local preview

```bash
python3 -m http.server 4173 --directory website
```

Open `http://localhost:4173`.

## Refresh the public graph index

Only run this command with a trusted pickle file. Python pickle loading can execute code.

```bash
python3 -m pip install networkx
python website/scripts/export_graph_index.py /path/to/final.pkl website/data/entities.json
```

The generated index contains only entity labels, resolved Wikidata QIDs, forward-edge object in-degree values, and aggregate graph metadata. It does not contain edge evidence, questions, training data, API tokens, or local source paths.

## GitHub Pages

The workflow at `.github/workflows/pages.yml` publishes this directory without a server. Sentence and dataset analysis happens entirely in the visitor's browser.
