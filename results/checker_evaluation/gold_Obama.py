# Reference labels, Obama (held-out test topic). key: (gold, topic_ids, confidence)
P1a, P1b, P1c = "Obama_prompt1_seed101|", "Obama_prompt1_seed202|", "Obama_prompt1_seed303|"
P2a, P2b, P2c = "Obama_prompt2_seed101|", "Obama_prompt2_seed202|", "Obama_prompt2_seed303|"
S, U, C = "SUPPORTED", "UNKNOWN", "CONTRADICTED"
GOLD = {
    P1a + "S01_C01": (U, [], "high"),     # proportion
    P1a + "S01_C02": (U, ["034"], "high"),  # puppet not in cited
    P1a + "S02_C01": (U, [], "low"),      # 061, 012 humorous; 008/044 not
    P1a + "S02_C02": (S, [], "high"),     # devil 008/044, Satan 061, Anti Christ 008
    P1a + "S03_C01": (U, ["029"], "low"),   # mockery only 085, 090
    P1a + "S04_C01": (S, [], "high"),     # 003, 020, 040, 058, 070
    P1a + "S04_C02": (S, [], "high"),     # 020, 040, 067, 070
    P1a + "S05_C01": (S, [], "high"),     # 081, 036
    P1a + "S05_C02": (U, ["074"], "low"),
    P1b + "S01_C01": (U, [], "high"),
    P1b + "S01_C02": (U, ["020", "033"], "high"),
    P1b + "S02_C01": (S, [], "high"),     # 008, 012, 044
    P1b + "S02_C02": (U, ["034", "084"], "high"),  # puppet, military not in cited
    P1b + "S03_C01": (U, ["067"], "high"),  # immigration not in cited
    P1b + "S03_C02": (S, [], "low"),      # 058 "show us some change"
    P1b + "S03_C03": (U, [], "high"),
    P1b + "S04_C01": (U, [], "high"),     # "significant"
    P1b + "S04_C02": (S, [], "high"),     # 021, 080
    P1b + "S04_C03": (S, [], "high"),     # 047, 080
    P1b + "S04_C04": (U, [], "high"),
    P1b + "S05_C01": (U, [], "high"),     # proportion
    P1c + "S01_C01": (U, [], "high"),
    P1c + "S01_C02": (S, [], "low"),      # tax 014, guns 003/040/070, health 020/058
    P1c + "S02_C01": (S, [], "high"),     # 044 attack, 008 conspiracy
    P1c + "S02_C02": (S, [], "high"),     # 008
    P1c + "S02_C03": (U, ["034"], "high"),
    P1c + "S03_C01": (S, [], "high"),     # 010, 024, 031, 051, 080
    P1c + "S03_C02": (U, [], "high"),
    P1c + "S04_C01": (S, [], "high"),
    P1c + "S04_C02": (S, [], "high"),     # 021 fashion, 052 demeanor
    P1c + "S05_C01": (U, [], "high"),     # proportion
    P1c + "S05_C02": (S, [], "low"),      # 013, 077, 023, 074
    P1c + "S05_C03": (S, [], "low"),      # 073
    P2a + "S01_C01": (S, [], "low"),      # 001, 005, 080
    P2a + "S01_C02": (U, [], "high"),     # nothing on Barack's leadership
    P2a + "S01_C03": (U, ["020", "033"], "high"),
    P2a + "S02_C01": (U, [], "low"),      # "the commentary focuses on" = share
    P2a + "S02_C02": (S, [], "high"),     # 003, 058
    P2a + "S02_C03": (U, ["040"], "low"),
    P2a + "S03_C01": (U, ["001"], "high"),  # 054 (+001) < 3
    P2a + "S03_C02": (U, ["033"], "low"),
    P2a + "S03_C03": (U, [], "low"),
    P2a + "S04_C01": (S, [], "high"),     # 008, 044, 061
    P2a + "S04_C02": (S, [], "high"),     # 044, 061
    P2a + "S04_C03": (U, ["011"], "low"),
    P2a + "S05_C01": (U, ["052", "080"], "high"),  # 021, 047 = 2
    P2a + "S05_C02": (U, ["080", "039"], "high"),
    P2b + "S01_C01": (S, [], "high"),     # all 90 cited
    P2b + "S01_C02": (U, [], "low"),      # appearance 081, no leadership style
    P2b + "S01_C03": (S, [], "high"),     # 020, 033, 087
    P2b + "S02_C01": (U, [], "high"),
    P2b + "S02_C02": (U, ["041", "081"], "high"),
    P2b + "S02_C03": (S, [], "high"),     # 021, 080, 052
    P2b + "S02_C04": (U, [], "low"),
    P2b + "S03_C01": (U, [], "high"),     # proportion
    P2b + "S03_C02": (U, ["034"], "high"),
    P2b + "S03_C03": (S, [], "high"),     # 008
    P2b + "S03_C04": (S, [], "high"),     # 033, 040, 087
    P2b + "S04_C01": (S, [], "low"),      # 003/070/088 guns, 014 tax, 020 health, 067 immigration
    P2b + "S04_C02": (U, [], "high"),
    P2b + "S05_C01": (S, [], "low"),      # 005, 010, 080, 083
    P2b + "S05_C02": (S, [], "low"),      # 031, 062
    P2b + "S05_C03": (U, [], "low"),
    P2c + "S01_C01": (U, [], "high"),
    P2c + "S01_C02": (U, ["023", "040"], "high"),  # spending not in cited
    P2c + "S02_C01": (U, [], "high"),     # proportion
    P2c + "S02_C02": (U, ["034"], "high"),
    P2c + "S02_C03": (S, [], "high"),     # 008, 044
    P2c + "S02_C04": (U, [], "low"),
    P2c + "S03_C01": (S, [], "high"),     # 021, 047, 052, 085
    P2c + "S03_C02": (S, [], "high"),     # 021, 052, 080
    P2c + "S03_C03": (U, [], "high"),     # "mostly" = proportion
    P2c + "S04_C01": (S, [], "high"),     # 013, 037, 043
    P2c + "S04_C02": (S, [], "high"),     # 031, 037, 043, 090
    P2c + "S05_C01": (S, [], "low"),      # 051 inauguration, 025 money, 050 interaction
    P2c + "S05_C02": (U, [], "low"),      # inferred attitude
}
