"""Score checker verdicts against the reference labels."""
import json, sys, collections, importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
def load_gold():
    gold = {}
    for topic in ("McDonalds", "Obama", "Superbowl"):
        spec = importlib.util.spec_from_file_location(topic, os.path.join(HERE, f"gold_{topic}.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        gold.update(mod.GOLD)
    return gold
def score(pred, gold, keys, label=""):
    keys = [k for k in keys if k in pred]
    cm = collections.Counter((gold[k][0], pred[k]) for k in keys)
    n = len(keys); acc = sum(gold[k][0] == pred[k] for k in keys) / n
    gs = [k for k in keys if gold[k][0] == "SUPPORTED"]; ps = [k for k in keys if pred[k] == "SUPPORTED"]
    tp = sum(pred[k] == "SUPPORTED" for k in gs)
    prec = tp / len(ps) if ps else float("nan"); rec = tp / len(gs) if gs else float("nan")
    unk = sum(pred[k] == "UNKNOWN" for k in keys) / n
    print(f"{label:28s} n={n:3d} acc={acc:.3f} SUPPORTED precision={prec:.3f} recall={rec:.3f} "
          f"| false UNKNOWN={sum(pred[k]=='UNKNOWN' for k in gs):3d} false SUPPORTED={len(ps)-tp:2d} "
          f"| UNKNOWN rate={unk:.2f} | CONTRADICTED={sum(pred[k]=='CONTRADICTED' for k in keys)}")
    return cm
if __name__ == "__main__":
    gold = load_gold()
    claims = json.load(open(os.path.join(HERE, "claims.json"), encoding="utf-8"))
    assert set(gold) == {c["key"] for c in claims}, set(gold) ^ {c["key"] for c in claims}
    print("gold:", collections.Counter(v[0] for v in gold.values()), collections.Counter(v[2] for v in gold.values()))
    pred = {c["key"]: c["baseline"] for c in claims}
    if len(sys.argv) > 1:
        pred = {json.loads(l)["key"]: json.loads(l)["verdict"] for l in open(sys.argv[1], encoding="utf-8")}
    for name, keys in (("dev (McDonalds)", [k for k in gold if k.startswith("McD")]),
                       ("test (Obama+Superbowl)", [k for k in gold if not k.startswith("McD")]),
                       ("all", list(gold)), ("all, high-confidence", [k for k in gold if gold[k][2] == "high"])):
        cm = score(pred, gold, keys, name)
    print("confusion (gold, predicted) on all:", dict(cm))
