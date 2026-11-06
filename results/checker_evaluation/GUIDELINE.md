# Reference labels for summary-claim verification

Unit: one extracted summary claim (the `claim` text), judged against the tweets
cited for it (`cited`). The summary_quote is context only. No outside knowledge.

gold (cited scope), one of SUPPORTED / CONTRADICTED / UNKNOWN:

1. A tweet supports the claim when a reasonable reader sees the same content in
   it, in other words is fine ("I really really want an ice cream cone" shows a
   strong craving). A category may name its instances (J Cole, Bob Marley ->
   cultural figures). Added motives, causes or judgments the tweet does not
   express are not support.
2. Count needed, from the claim's own wording:
   - "some", "others", a bare plural ("users", "people", "posts"), "there are",
     "include", "is described", or no quantifier: 1 supporting tweet.
   - "several", "multiple", "many", "numerous", "frequently", "often",
     "recurring", "a number of", "common": 3 distinct supporting tweets.
   - proportion of the whole collection ("most", "majority", "a significant /
     large portion or number", "a minority", "generally", "overall", "the user
     base"): UNKNOWN, cited tweets cannot establish a share of the corpus.
   - "mixed", "diverse", "a range / spectrum / variety of reactions": supported
     by cited tweets showing at least two clearly different reactions (three
     for "wide range" or "variety").
3. Lists (A, B and C; A, B or C; "items like A and B"): every listed detail
   needs support, since each is a statement the summary makes about the tweets.
   A count word applies to the claim as a whole ("tweets frequently reference
   A, B and C": 3 tweets in total, each item at least once).
4. Other cited tweets that say something else never count against an
   existential claim ("some users like X" stays supported when others hate X).
5. CONTRADICTED only when a cited tweet explicitly states the opposite of the
   claim itself (not of another user's view). Missing content is UNKNOWN.

topic_ids (for claims not SUPPORTED by their citations): the uncited tweets of
the same topic that would support the claim under the same rules; empty if none.
