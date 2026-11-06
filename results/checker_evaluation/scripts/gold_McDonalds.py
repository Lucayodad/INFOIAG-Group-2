# Reference labels, McDonalds (development topic). key: (gold, topic_ids, confidence)
# gold = verdict from the cited tweets under GUIDELINE.md; topic_ids = uncited
# tweets of the topic that would support a non-SUPPORTED claim.
P1a, P1b, P1c = "McDonalds_prompt1_seed101|", "McDonalds_prompt1_seed202|", "McDonalds_prompt1_seed303|"
P2a, P2b, P2c = "McDonalds_prompt2_seed101|", "McDonalds_prompt2_seed202|", "McDonalds_prompt2_seed303|"
S, U, C = "SUPPORTED", "UNKNOWN", "CONTRADICTED"
GOLD = {
    P1a + "S01_C01": (S, [], "high"),     # 003, 026, 051 want / go straight to McDonald's
    P1a + "S01_C02": (U, ["015"], "high"),  # McRib not in cited
    P1a + "S02_C01": (U, [], "high"),     # proportion
    P1a + "S02_C02": (S, [], "high"),     # sauce 020/078, line too long 076
    P1a + "S03_C01": (S, [], "low"),      # 010, 045, 057 going with others
    P1a + "S03_C02": (U, ["066"], "low"),  # sharing meals not shown
    P1a + "S04_C01": (U, [], "high"),     # no location confusion
    P1a + "S05_C01": (S, [], "high"),     # 001 delicious
    P1a + "S05_C02": (S, [], "high"),     # 009, 046, 056
    P1a + "S05_C03": (S, [], "high"),     # 056 disgusting
    P1b + "S01_C01": (U, [], "high"),     # proportion
    P1b + "S01_C02": (U, ["042", "018", "054"], "high"),  # coffee missing; only 2 item desires
    P1b + "S02_C01": (U, [], "high"),     # comfort missing
    P1b + "S02_C02": (S, [], "high"),     # 056, 071
    P1b + "S02_C03": (S, [], "high"),     # 046
    P1b + "S03_C01": (U, [], "low"),      # <3 ordering frustrations
    P1b + "S03_C02": (S, [], "high"),     # 076
    P1b + "S03_C03": (U, ["020", "078"], "high"),  # sauce not in cited
    P1b + "S03_C04": (U, ["024", "025"], "high"),  # 034 is a customer
    P1b + "S04_C01": (U, ["015", "029", "043", "077"], "low"),
    P1b + "S04_C02": (U, [], "high"),
    P1b + "S04_C03": (U, ["015"], "low"),
    P1b + "S05_C01": (U, [], "high"),     # 044 says McDonald's is cheaper
    P1b + "S05_C02": (S, [], "high"),     # 044
    P1c + "S01_C01": (S, [], "high"),     # diverse: craving, disgust, complaints
    P1c + "S01_C02": (U, ["026"], "low"),   # nugget craving not in cited
    P1c + "S01_C03": (S, [], "high"),     # 004/072 food, 024/025 service
    P1c + "S02_C01": (S, [], "high"),     # 002, 058, 078, 035
    P1c + "S02_C02": (U, [], "high"),     # 002, 058 = 2
    P1c + "S02_C03": (U, ["020"], "high"),  # 078 only (+020 uncited = 2)
    P1c + "S02_C04": (U, [], "high"),
    P1c + "S03_C01": (S, [], "high"),     # 004, 056, 071 taste
    P1c + "S03_C02": (S, [], "high"),     # 056, 071
    P1c + "S04_C01": (U, [], "high"),     # proportion
    P1c + "S05_C01": (S, [], "high"),     # 053 employee interaction
    P1c + "S05_C02": (U, ["077"], "high"),  # card machine story uncited
    P1c + "S05_C03": (S, [], "high"),     # 082, 084 compare chains
    P2a + "S01_C01": (S, [], "high"),
    P2a + "S01_C02": (S, [], "high"),     # 003
    P2a + "S01_C03": (U, [], "low"),
    P2a + "S01_C04": (S, [], "low"),      # 004 "done got Nasty"
    P2a + "S02_C01": (S, [], "high"),
    P2a + "S02_C02": (U, [], "high"),     # "salty" in no tweet
    P2a + "S03_C01": (S, [], "high"),     # sauce 020/078, hours 058
    P2a + "S03_C02": (S, [], "high"),
    P2a + "S03_C03": (S, [], "high"),     # 058
    P2a + "S04_C01": (S, [], "high"),     # 024, 025
    P2a + "S04_C02": (S, [], "high"),
    P2a + "S05_C01": (U, ["021", "045", "017"], "high"),  # only 060 cited
    P2a + "S05_C02": (U, [], "high"),
    P2b + "S01_C01": (S, [], "high"),     # positive and negative
    P2b + "S01_C02": (U, ["015", "026"], "high"),
    P2b + "S01_C03": (U, [], "high"),     # no pricing complaint
    P2b + "S02_C01": (U, [], "high"),
    P2b + "S02_C02": (S, [], "high"),     # 020
    P2b + "S02_C03": (U, [], "low"),      # 076 not a drive-thru
    P2b + "S03_C01": (S, [], "high"),
    P2b + "S03_C02": (S, [], "high"),     # 071, 072
    P2b + "S04_C01": (S, [], "high"),     # 003, 026, 047, 054
    P2b + "S04_C02": (U, ["083", "085", "052", "020"], "high"),  # breakfast items not in cited
    P2b + "S05_C01": (U, ["044", "084"], "high"),  # 082 only
    P2b + "S05_C02": (S, [], "low"),      # 019 commercial annoying
    P2b + "S05_C03": (U, [], "low"),      # catchy is not attractive
    P2c + "S01_C01": (S, [], "high"),
    P2c + "S01_C02": (U, ["026"], "high"),
    P2c + "S01_C03": (U, ["072"], "high"),  # stale fish uncited
    P2c + "S02_C01": (S, [], "high"),     # 013, 058, 076
    P2c + "S02_C02": (U, [], "low"),
    P2c + "S02_C03": (S, [], "high"),     # 058
    P2c + "S02_C04": (U, ["064"], "high"),
    P2c + "S03_C01": (U, ["009", "046"], "high"),
    P2c + "S03_C02": (S, [], "high"),     # 063
    P2c + "S03_C03": (U, ["045", "017"], "low"),
    P2c + "S04_C01": (U, [], "low"),      # 015, 021 = 2
    P2c + "S04_C02": (U, ["060"], "high"),
    P2c + "S05_C01": (U, ["077"], "high"),  # payment missing
    P2c + "S05_C02": (U, ["087"], "high"),
    P2c + "S05_C03": (S, [], "low"),      # 033 other side of the island
}
