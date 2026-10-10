# Ontology input

`group2_Ontology.owl` is the user's supplied RDF/XML export, preserved byte-for-byte.
The namespace is `http://www.semanticweb.org/group2/ontology#`.
The current loader accepts RDF/XML. It reads a local file and does not fetch imports.
Its in-memory inference and temporary candidate checks do not change this file.
See `../docs/SECTION_2.md` for the exact supported reasoning scope.
