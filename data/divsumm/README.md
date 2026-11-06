# DivSumm: Obama, Super Bowl and McDonald's

The input tweets of the three DivSumm topics this project uses: 90 tweets per
topic, 30 for each dialect group (AA, Hisp, White).

`ready_to_use/<topic>.jsonl` holds one JSON object per tweet, in the original
order, with the fields `id`, `topic`, `dialect_label`, `text`, `source_file`
and `source_line`. The `id` is generated from the topic and the original line
number; it is not a Twitter ID.

`ready_to_use/references.json` holds DivSumm's human-written summaries: two
abstractive and two extractive per topic. They are never shown to the model;
`evaluate.py` scores the generated summaries against the abstractive ones
(ROUGE, BARTScore).

The dialect labels are the dataset's own (African-American, Hispanic, White
dialect), not verified identities of the authors. In this project they are read
only by the final evaluation, never by the agent.

No tweet was removed or rewritten: spelling, dialect, offensive wording, HTML
entities such as `&amp;` and literal backslash-n sequences are preserved. Files
of the original archive that were not valid UTF-8 were decoded as Mac Roman.

## Source

Repository: https://github.com/PortNLP/DivSumm

Olabisi, Olubusayo; Hudson, Aaron; Jetter, Antonie; Agrawal, Ameeta. "Analyzing
the Dialect Diversity in Multi-document Summaries." COLING 2022.
https://aclanthology.org/2022.coling-1.542/
