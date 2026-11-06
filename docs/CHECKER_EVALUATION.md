# Fact-checker evaluation and revision (2026-11-05)

The fact checker is the instrument behind every factual number in the report
and the only trigger for a revision (a CONTRADICTED claim). In the label-free
final run (results/agent_compliant_v2), 161 of the 210 claims of the 18 drafts
(77%) were UNKNOWN. UNKNOWN is the open-world answer when the cited tweets
neither establish nor contradict a claim, so a high rate is not by itself an
error. We measured how many verdicts were wrong, why, and revised the checker.
All materials are in results/checker_evaluation/.

## Reference labels
All 210 claims were labeled against their cited tweets under a written guideline
(results/checker_evaluation/GUIDELINE.md): support may be paraphrased; "some",
"others", a plural or no count word needs one supporting tweet; "several",
"many", "frequently", "often" need three; a share of the whole collection
("most", "a significant portion") cannot be established from cited tweets;
every listed detail needs support; other users' opinions never contradict
"some users say X"; missing content is never a contradiction. Result: 94
SUPPORTED, 116 UNKNOWN, 0 CONTRADICTED; 54 marked low confidence. For each claim
not supported by its citations, the uncited tweets of the same topic that would
support it were recorded.

The labels were produced by Claude (an LLM), independently of Qwen's verdicts,
and are not yet validated by people. reference_labels.csv marks a random sample
of 60 claims for two team members to label independently; their agreement with
the reference (Cohen's kappa) must be reported with these results.

## Protocol
- Development topic McDonalds (78 claims): every design decision.
- Held-out topics Obama and Superbowl (132 claims): run once, with the design
  frozen beforehand.
- During tuning the agent's first turn, which was lookup_fact in all 210 recorded
  checks, is replayed from the record; every verdict is a live Qwen call with the
  production settings (temperature 0, seed 42).
- Metrics: accuracy; precision and recall of SUPPORTED; false UNKNOWN (a
  supported claim not confirmed); false SUPPORTED; false CONTRADICTED.

## Findings on the original checker
- Accuracy 0.67, precision 0.78, recall 0.38: 57 supported claims were UNKNOWN,
  10 unsupported claims SUPPORTED, and all 3 CONTRADICTED verdicts were missing
  information, not an opposing tweet. These 3 are the contradictions repaired in
  agent_compliant_v2 ("CONTRADICTED 3 -> 0"), so that result does not show that
  the agent removed factual errors.
- 13 checks never reached a verdict: 10 requests exceeded a self-imposed 16 KB
  cap (the answer schema repeats every evidence quote), 3 answers hit the
  900-token limit.
- At most 16 evidence items (tweets and their extracted interpretations, ranked
  by word overlap) were shown: claims citing 10 or more tweets often lost the
  tweets that showed them.
- The model applied its own rules inconsistently: it asked three tweets for
  "some"/"others", let other users' opinions block "some users say X", and
  accepted lists with a missing item.
- 58 of the 116 claims not supported by their citations are supported by an
  uncited tweet of the same topic: the summary cited the wrong tweets. No checker
  can confirm these from the citations; this needs citation repair (below).
- Not reproducible across sessions: a byte-identical request returned a
  different verdict in 2 of 5 cases (temperature 0, fixed seed). Within one run
  identical requests repeat (prompt cache), across runs they can differ.

## Designs tested (development topic, 78 claims)
| Design | Accuracy | Precision | Recall | False UNKNOWN | False SUPPORTED |
|---|---|---|---|---|---|
| Original checker (recorded) | 0.63 | 0.74 | 0.38 | 23 | 5 |
| A: explicit rules, 32 KB cap, 1500 answer tokens | 0.69 | 0.64 | 0.78 | 8 | 16 |
| B: A + explanation and quotes before the verdict | 0.76 | 0.82 | 0.62 | 14 | 5 |
| D: model reports content, code applies counting rules | 0.69 | 0.88 | 0.41 | 22 | 2 |
| C: B + every cited tweet shown before any interpretation | 0.81 | 0.84 | 0.73 | 9 | 5 |
| **Final: C + two rules in code** | **0.82** | **0.87** | **0.73** | **10** | **4** |

- A traded strictness for leniency: most new false SUPPORTED are lists with one
  unsupported item.
- B: writing the explanation and choosing quotations before the verdict field
  (the JSON field order of constrained decoding) let the model check each item.
- D failed in an instructive way: asked to list the claim details no cited tweet
  shows, the 4B model rarely leaves the list empty; it lists words of the claim
  itself ("frequently", "rude"), tweet IDs, and twice "McRibs" copied from the
  instruction's example. Re-deriving all 78 verdicts offline from the saved
  reports reproduces them, so the counting code was right and the model's
  structured report was not.
- C: showing every cited tweet first raised recall on the 18 claims citing 10 or
  more tweets from 0.33 (B) to 0.67.

Final design, frozen on the development topic before the held-out run: C plus
two rules applied by code after the model's verdict:
1. a claim stating a share of the whole collection ("most", "a significant
   number") is never SUPPORTED, and a claim with "many", "several",
   "frequently"... is SUPPORTED only when its references quote three distinct
   tweets;
2. a claim about what users say or what tweets contain is never CONTRADICTED
   (other users saying something else does not refute it): UNKNOWN instead.
On the development topic the rules changed two verdicts, both toward the label.

## Held-out result (Obama + Superbowl, 132 claims, one run)
| | Accuracy | Precision | Recall | False UNKNOWN | False SUPPORTED | CONTRADICTED |
|---|---|---|---|---|---|---|
| Original checker | 0.69 | 0.82 | 0.39 | 34 | 5 | 2 (both false) |
| Final design | **0.80** | **0.90** | **0.61** | **22** | **4** | 1 (false) |
| Original, high-confidence labels (93) | 0.71 | 0.84 | 0.49 | 22 | 4 | 1 |
| Final, high-confidence labels (93) | **0.84** | **0.94** | **0.72** | **12** | **2** | 1 |

Per topic: Obama 0.70 -> 0.80, Superbowl 0.68 -> 0.79. No check failed for
technical reasons. Paired on the 132 held-out claims, the final design corrects
21 verdicts and breaks 7 (exact McNemar p = 0.0125); this includes Qwen's
run-to-run noise, since the original verdicts come from an earlier session.

## Limitations
- Recall 0.61: about 4 in 10 supported claims are still not confirmed.
- One false contradiction remains ("There are positive observations about Barack
  Obama's speeches and appearance"); the attribution rule's word list does not
  cover this phrasing. Widening it after seeing the test would leak the test.
- Wrong citations stay UNKNOWN. A citation-repair step is feasible: ranking the
  topic's uncited tweets by embedding similarity plus idf-weighted word overlap
  puts a supporting tweet in the top 5 for 38 of the 58 miscited claims (66%);
  it is not yet implemented in the agent.
- The reference labels are one annotator's (an LLM's) until the human sample is
  labeled.
- Run-to-run variation of Qwen remains; results come from single runs.

## Rerun of the 18 trials with the revised checker (2026-11-06)
`results/agent_checker_v2` (label-free agent, default settings; every claim
checked anew, claim extraction reused) and `results/agent_checker_v2_proportion`
(the same with `--proportion-check`, seeded with the first run's fresh checks).
Both evaluated with `evaluate_label_free_run.py` (labels only at that step).

| Drafts (210 claims) | Original checker | Revised checker |
|---|---|---|
| SUPPORTED | 46 (22%) | 83 (40%) |
| CONTRADICTED | 3 (all false) | 0 |
| UNKNOWN | 161 (77%) | 127 (60%) |

- Main run: no claim of any draft is contradicted, so the agent, which revises
  only contradicted claims, changes 0/18 summaries; mean Equal Coverage 0.0024.
  The earlier "CONTRADICTED 3 -> 0" improvement came from checker errors.
- Proportion feedback layer: 5 drafts judged proportional; 9 flags were found
  already reported by a verified claim (no rewrite); 2 feedbacks invalid; 2 real
  gaps. Superbowl P1 s303: three rewrites rolled back (not supported by the
  checker). Superbowl P2 s202: one rewrite accepted (SUPPORTED 1 -> 2), but by
  the labels it made coverage less equal: Equal Coverage 0.0032 -> 0.0033, worst
  group 2.76% -> 1.90%, four-fifths disparity 0.22 -> 0.29. Across the 18 trials
  mean Equal Coverage stays 0.0024. The improvement seen in the earlier smoke
  trial (Superbowl P1 s101) did not recur: with the revised checker that
  draft's flagged opinion is already reported by a verified claim.

## In the code
fact_check.py and fact_agent.py (checker version 2026-11-05, stamped on every
assessment): the instructions of design C, the verdict as the last answer field,
every cited tweet before any interpretation (up to 40 items / 12000 bytes),
40000-byte requests and 1500 answer tokens, and counting_rule() in
apply_verdict. Recorded verdicts of the old checker are never reused
(summary_agent.seed_check_cache); claim extractions are. tests/test_checker_rules.py
covers the rules, the evidence order, the field order, the limits and the
version gate. The repository code sends byte-identical requests to the tested
design on all 210 claims.

Reproduce the held-out score: in results/checker_evaluation/scripts, score
runs/test_C.jsonl with guard.final_guard against the gold_*.py labels (score.py).
