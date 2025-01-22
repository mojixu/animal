def DOG_mapping():
    class_names = {
        0: "Chihuahua",
        1: "Japanese_spaniel",
        2: "Maltese_dog",
        3: "Pekinese",
        4: "Shih-Tzu",
        5: "Blenheim_spaniel",
        6: "papillon",
        7: "toy_terrier",
        8: "Rhodesian_ridgeback",
        9: "Afghan_hound",
        10: "basset",
        11: "beagle",
        12: "bloodhound",
        13: "bluetick",
        14: "black-and-tan_coonhound",
        15: "Walker_hound",
        16: "English_foxhound",
        17: "redbone",
        18: "borzoi",
        19: "Irish_wolfhound",
        20: "Italian_greyhound",
        21: "whippet",
        22: "Ibizan_hound",
        23: "Norwegian_elkhound",
        24: "otterhound",
        25: "Saluki",
        26: "Scottish_deerhound",
        27: "Weimaraner",
        28: "Staffordshire_bullterrier",
        29: "American_Staffordshire_terrier",
        30: "Bedlington_terrier",
        31: "Border_terrier",
        32: "Kerry_blue_terrier",
        33: "Irish_terrier",
        34: "Norfolk_terrier",
        35: "Norwich_terrier",
        36: "Yorkshire_terrier",
        37: "wire-haired_fox_terrier",
        38: "Lakeland_terrier",
        39: "Sealyham_terrier",
        40: "Airedale",
        41: "cairn",
        42: "Australian_terrier",
        43: "Dandie_Dinmont",
        44: "Boston_bull",
        45: "miniature_schnauzer",
        46: "giant_schnauzer",
        47: "standard_schnauzer",
        48: "Scotch_terrier",
        49: "Tibetan_terrier",
        50: "silky_terrier",
        51: "soft-coated_wheaten_terrier",
        52: "West_Highland_white_terrier",
        53: "Lhasa",
        54: "flat-coated_retriever",
        55: "curly-coated_retriever",
        56: "golden_retriever",
        57: "Labrador_retriever",
        58: "Chesapeake_Bay_retriever",
        59: "German_short-haired_pointer",
        60: "vizsla",
        61: "English_setter",
        62: "Irish_setter",
        63: "Gordon_setter",
        64: "Brittany_spaniel",
        65: "clumber",
        66: "English_springer",
        67: "Welsh_springer_spaniel",
        68: "cocker_spaniel",
        69: "Sussex_spaniel",
        70: "Irish_water_spaniel",
        71: "kuvasz",
        72: "schipperke",
        73: "groenendael",
        74: "malinois",
        75: "briard",
        76: "kelpie",
        77: "komondor",
        78: "Old_English_sheepdog",
        79: "Shetland_sheepdog",
        80: "collie",
        81: "Border_collie",
        82: "Bouvier_des_Flandres",
        83: "Rottweiler",
        84: "German_shepherd",
        85: "Doberman",
        86: "miniature_pinscher",
        87: "Greater_Swiss_Mountain_dog",
        88: "Bernese_mountain_dog",
        89: "Appenzeller",
        90: "EntleBucher",
        91: "boxer",
        92: "bull_mastiff",
        93: "Tibetan_mastiff",
        94: "French_bulldog",
        95: "Great_Dane",
        96: "Saint_Bernard",
        97: "Eskimo_dog",
        98: "malamute",
        99: "Siberian_husky",
        100: "affenpinscher",
        101: "basenji",
        102: "pug",
        103: "Leonberg",
        104: "Newfoundland",
        105: "Great_Pyrenees",
        106: "Samoyed",
        107: "Pomeranian",
        108: "chow",
        109: "keeshond",
        110: "Brabancon_griffon",
        111: "Pembroke",
        112: "Cardigan",
        113: "toy_poodle",
        114: "miniature_poodle",
        115: "standard_poodle",
        116: "Mexican_hairless",
        117: "dingo",
        118: "dhole",
        119: "African_hunting_dog"
    }
    return class_names

def CUB_mapping():

    class_names = {
    0: "Black_footed_Albatross",
    1: "Laysan_Albatross",
    2: "Sooty_Albatross",
    3: "Groove_billed_Ani",
    4: "Crested_Auklet",
    5: "Least_Auklet",
    6: "Parakeet_Auklet",
    7: "Rhinoceros_Auklet",
    8: "Brewer_Blackbird",
    9: "Red_winged_Blackbird",
    10: "Rusty_Blackbird",
    11: "Yellow_headed_Blackbird",
    12: "Bobolink",
    13: "Indigo_Bunting",
    14: "Lazuli_Bunting",
    15: "Painted_Bunting",
    16: "Cardinal",
    17: "Spotted_Catbird",
    18: "Gray_Catbird",
    19: "Yellow_breasted_Chat",
    20: "Eastern_Towhee",
    21: "Chuck_will_Widow",
    22: "Brandt_Cormorant",
    23: "Red_faced_Cormorant",
    24: "Pelagic_Cormorant",
    25: "Bronzed_Cowbird",
    26: "Shiny_Cowbird",
    27: "Brown_Creeper",
    28: "American_Crow",
    29: "Fish_Crow",
    30: "Black_billed_Cuckoo",
    31: "Mangrove_Cuckoo",
    32: "Yellow_billed_Cuckoo",
    33: "Gray_crowned_Rosy_Finch",
    34: "Purple_Finch",
    35: "Northern_Flicker",
    36: "Acadian_Flycatcher",
    37: "Great_Crested_Flycatcher",
    38: "Least_Flycatcher",
    39: "Olive_sided_Flycatcher",
    40: "Scissor_tailed_Flycatcher",
    41: "Vermilion_Flycatcher",
    42: "Yellow_bellied_Flycatcher",
    43: "Frigatebird",
    44: "Northern_Fulmar",
    45: "Gadwall",
    46: "American_Goldfinch",
    47: "European_Goldfinch",
    48: "Boat_tailed_Grackle",
    49: "Eared_Grebe",
    50: "Horned_Grebe",
    51: "Pied_billed_Grebe",
    52: "Western_Grebe",
    53: "Blue_Grosbeak",
    54: "Evening_Grosbeak",
    55: "Pine_Grosbeak",
    56: "Rose_breasted_Grosbeak",
    57: "Pigeon_Guillemot",
    58: "California_Gull",
    59: "Glaucous_winged_Gull",
    60: "Heermann_Gull",
    61: "Herring_Gull",
    62: "Ivory_Gull",
    63: "Ring_billed_Gull",
    64: "Slaty_backed_Gull",
    65: "Western_Gull",
    66: "Anna_Hummingbird",
    67: "Ruby_throated_Hummingbird",
    68: "Rufous_Hummingbird",
    69: "Green_Violetear",
    70: "Long_tailed_Jaeger",
    71: "Pomarine_Jaeger",
    72: "Blue_Jay",
    73: "Florida_Jay",
    74: "Green_Jay",
    75: "Dark_eyed_Junco",
    76: "Tropical_Kingbird",
    77: "Gray_Kingbird",
    78: "Belted_Kingfisher",
    79: "Green_Kingfisher",
    80: "Pied_Kingfisher",
    81: "Ringed_Kingfisher",
    82: "White_breasted_Kingfisher",
    83: "Red_legged_Kittiwake",
    84: "Horned_Lark",
    85: "Pacific_Loon",
    86: "Mallard",
    87: "Western_Meadowlark",
    88: "Hooded_Merganser",
    89: "Red_breasted_Merganser",
    90: "Mockingbird",
    91: "Nighthawk",
    92: "Clark_Nutcracker",
    93: "White_breasted_Nuthatch",
    94: "Baltimore_Oriole",
    95: "Hooded_Oriole",
    96: "Orchard_Oriole",
    97: "Scott_Oriole",
    98: "Ovenbird",
    99: "Brown_Pelican",
    100: "White_Pelican",
    101: "Western_Wood_Pewee",
    102: "Sayornis",
    103: "American_Pipit",
    104: "Whip_poor_Will",
    105: "Horned_Puffin",
    106: "Common_Raven",
    107: "White_necked_Raven",
    108: "American_Redstart",
    109: "Geococcyx",
    110: "Loggerhead_Shrike",
    111: "Great_Grey_Shrike",
    112: "Baird_Sparrow",
    113: "Black_throated_Sparrow",
    114: "Brewer_Sparrow",
    115: "Chipping_Sparrow",
    116: "Clay_colored_Sparrow",
    117: "House_Sparrow",
    118: "Field_Sparrow",
    119: "Fox_Sparrow",
    120: "Grasshopper_Sparrow",
    121: "Harris_Sparrow",
    122: "Henslow_Sparrow",
    123: "Le_Conte_Sparrow",
    124: "Lincoln_Sparrow",
    125: "Nelson_Sharp_tailed_Sparrow",
    126: "Savannah_Sparrow",
    127: "Seaside_Sparrow",
    128: "Song_Sparrow",
    129: "Tree_Sparrow",
    130: "Vesper_Sparrow",
    131: "White_crowned_Sparrow",
    132: "White_throated_Sparrow",
    133: "Cape_Glossy_Starling",
    134: "Bank_Swallow",
    135: "Barn_Swallow",
    136: "Cliff_Swallow",
    137: "Tree_Swallow",
    138: "Scarlet_Tanager",
    139: "Summer_Tanager",
    140: "Artic_Tern",
    141: "Black_Tern",
    142: "Caspian_Tern",
    143: "Common_Tern",
    144: "Elegant_Tern",
    145: "Forsters_Tern",
    146: "Least_Tern",
    147: "Green_tailed_Towhee",
    148: "Brown_Thrasher",
    149: "Sage_Thrasher",
    150: "Black_capped_Vireo",
    151: "Blue_headed_Vireo",
    152: "Philadelphia_Vireo",
    153: "Red_eyed_Vireo",
    154: "Warbling_Vireo",
    155: "White_eyed_Vireo",
    156: "Yellow_throated_Vireo",
    157: "Bay_breasted_Warbler",
    158: "Black_and_white_Warbler",
    159: "Black_throated_Blue_Warbler",
    160: "Blue_winged_Warbler",
    161: "Canada_Warbler",
    162: "Cape_May_Warbler",
    163: "Cerulean_Warbler",
    164: "Chestnut_sided_Warbler",
    165: "Golden_winged_Warbler",
    166: "Hooded_Warbler",
    167: "Kentucky_Warbler",
    168: "Magnolia_Warbler",
    169: "Mourning_Warbler",
    170: "Myrtle_Warbler",
    171: "Nashville_Warbler",
    172: "Orange_crowned_Warbler",
    173: "Palm_Warbler",
    174: "Pine_Warbler",
    175: "Prairie_Warbler",
    176: "Prothonotary_Warbler",
    177: "Swainson_Warbler",
    178: "Tennessee_Warbler",
    179: "Wilson_Warbler",
    180: "Worm_eating_Warbler",
    181: "Yellow_Warbler",
    182: "Northern_Waterthrush",
    183: "Louisiana_Waterthrush",
    184: "Bohemian_Waxwing",
    185: "Cedar_Waxwing",
    186: "American_Three_toed_Woodpecker",
    187: "Pileated_Woodpecker",
    188: "Red_bellied_Woodpecker",
    189: "Red_cockaded_Woodpecker",
    190: "Red_headed_Woodpecker",
    191: "Downy_Woodpecker",
    192: "Bewick_Wren",
    193: "Cactus_Wren",
    194: "Carolina_Wren",
    195: "House_Wren",
    196: "Marsh_Wren",
    197: "Rock_Wren",
    198: "Winter_Wren",
    199: "Common_Yellowthroat"
}

    return class_names