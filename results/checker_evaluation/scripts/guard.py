"""Deterministic count rule applied after the model's verdict."""
import re
SHARE = re.compile(r"\b(most|majority|significant (?:portion|number|share)|large (?:portion|number|share)|"
                   r"minority|mostly|generally|overall|user base)\b", re.I)
MANY = re.compile(r"\b(several|multiple|many|numerous|frequently|often|recurring|a number of)\b", re.I)
TWEET = re.compile(r"/(?:tweet|claim)/([A-Za-z]+_\d+)(?:_C\d+)?$")

def cited_tweets(references):
    return {m.group(1) for r in references for m in [TWEET.search(r["evidence_id"])] if m}

def guard(claim_text, verdict, references):
    if verdict != "SUPPORTED":
        return verdict, None
    if SHARE.search(claim_text):
        return "UNKNOWN", "share_of_collection"
    if MANY.search(claim_text) and len(cited_tweets(references)) < 3:
        return "UNKNOWN", "fewer_than_three_tweets"
    return verdict, None

# Claims about what authors say or what tweets contain: other users saying something else never
# refutes them (open world), so the model's CONTRADICTED becomes UNKNOWN.
ABOUT_AUTHORS = re.compile(r"\b(some|others?|users?|people|many|several|tweets?|posts?|participants?|individuals?|"
                           r"contributors?|reactions?|commentary|comments?|fans?|respondents?|collection|messages?|"
                           r"remarks?|discussion|conversation)\b", re.I)


def final_guard(claim_text, verdict, references):
    """Design frozen on the development topic before the held-out test was scored."""
    if verdict == "CONTRADICTED" and ABOUT_AUTHORS.search(claim_text):
        return "UNKNOWN", "attributed_claim_not_refuted_by_other_users"
    return guard(claim_text, verdict, references)
