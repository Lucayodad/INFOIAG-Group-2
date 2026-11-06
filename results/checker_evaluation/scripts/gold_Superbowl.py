# Reference labels, Superbowl (held-out test topic). key: (gold, topic_ids, confidence)
P1a, P1b, P1c = "Superbowl_prompt1_seed101|", "Superbowl_prompt1_seed202|", "Superbowl_prompt1_seed303|"
P2a, P2b, P2c = "Superbowl_prompt2_seed101|", "Superbowl_prompt2_seed202|", "Superbowl_prompt2_seed303|"
S, U, C = "SUPPORTED", "UNKNOWN", "CONTRADICTED"
GOLD = {
    P1a + "S01_C01": (S, [], "high"),
    P1a + "S01_C02": (S, [], "high"),     # 008, 009, 017
    P1a + "S01_C03": (U, ["078", "039"], "low"),
    P1a + "S01_C04": (S, [], "low"),      # 015 Nicki Minaj / Beyonce on the Super Bowl
    P1a + "S02_C01": (S, [], "high"),     # 022, 024, 026, 038
    P1a + "S02_C02": (U, ["009"], "high"),  # Broncos not in 021-040
    P1a + "S03_C01": (S, [], "low"),      # 053, 056, 059
    P1a + "S03_C02": (U, ["015", "065"], "high"),
    P1a + "S04_C01": (U, [], "low"),
    P1a + "S04_C02": (U, [], "high"),
    P1a + "S05_C01": (U, [], "low"),      # no logistics
    P1a + "S05_C02": (U, ["071", "018"], "high"),
    P1b + "S01_C01": (S, [], "high"),     # 008, 020, 038, 057
    P1b + "S02_C01": (U, ["077", "027", "047"], "high"),  # cited tweets are all disinterest
    P1b + "S02_C02": (U, [], "high"),     # proportion
    P1b + "S04_C01": (S, [], "low"),      # 028 rules + history, 080 food
    P1b + "S05_C01": (S, [], "high"),     # 009, 017, 052; 011, 065
    P1b + "S05_C02": (S, [], "high"),     # 009 vs 052 vs 065
    P1c + "S01_C01": (S, [], "high"),     # all 89 cited
    P1c + "S01_C02": (S, [], "high"),     # 006, 008, 041, 052, 057
    P1c + "S02_C01": (U, [], "high"),     # proportion
    P1c + "S02_C02": (U, [], "high"),
    P1c + "S02_C03": (U, [], "low"),
    P1c + "S03_C01": (U, ["075"], "high"),  # Rihanna not in cited
    P1c + "S04_C01": (S, [], "high"),     # 018
    P1c + "S04_C02": (S, [], "high"),     # 028, 072
    P1c + "S05_C01": (U, ["047"], "low"),
    P1c + "S05_C02": (U, ["012"], "high"),
    P2a + "S01_C01": (U, [], "high"),     # "generally" = share
    P2a + "S01_C02": (S, [], "high"),     # 008, 020, 038, 057
    P2a + "S02_C01": (U, [], "high"),     # proportion
    P2a + "S02_C02": (U, ["078"], "low"),
    P2a + "S03_C01": (S, [], "high"),     # 018 commercials, 036/053/071 halftime
    P2a + "S03_C02": (S, [], "high"),     # 036, 053, 071
    P2a + "S04_C01": (S, [], "high"),     # 002, 010, 034, 051, 062
    P2a + "S04_C02": (U, ["054", "044"], "low"),
    P2a + "S05_C01": (S, [], "high"),     # 039, 048, 070, 078
    P2a + "S05_C02": (S, [], "low"),      # 061 Ravens vs 085 Patriots
    P2b + "S01_C01": (U, [], "low"),      # "highly polarized" not shown
    P2b + "S02_C01": (U, [], "low"),      # "the discussion focuses" = share
    P2b + "S03_C01": (U, [], "high"),     # 018 only; 060 positive
    P2b + "S03_C02": (S, [], "low"),      # 018 "SUCKED"
    P2b + "S05_C01": (U, [], "high"),     # "the user base"
    P2b + "S05_C02": (U, ["031", "034", "051"], "high"),
    P2c + "S01_C01": (S, [], "high"),     # 001-005 differ
    P2c + "S01_C02": (U, ["047", "074"], "high"),  # 49ers not in cited
    P2c + "S01_C03": (S, [], "high"),     # 002
    P2c + "S02_C01": (S, [], "high"),     # 006, 008
    P2c + "S02_C02": (U, ["026", "077"], "high"),  # 006, 008 = 2
    P2c + "S02_C03": (U, ["016", "083"], "low"),
    P2c + "S03_C01": (U, ["027", "030", "053", "056", "071"], "high"),
    P2c + "S03_C02": (U, ["056", "027"], "high"),
    P2c + "S04_C01": (U, ["018"], "high"),
    P2c + "S04_C02": (U, ["028", "072"], "high"),
    P2c + "S05_C01": (U, [], "low"),
    P2c + "S05_C02": (U, [], "low"),
}
