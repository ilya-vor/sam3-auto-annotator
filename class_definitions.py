# Class names and IDs
CLASS_NAMES_ALL = [
    "safety gloves",
    "safety shoes",
    "Welding mask",
    "hand",
    "shoes",
    "face"
]

# Target class IDs (last three)
BARE_HAND_ID = 3
FOOT_WITHOUT_SHOES_ID = 4
HUMAN_FACE_ID = 5

TARGET_IDS = {BARE_HAND_ID, FOOT_WITHOUT_SHOES_ID, HUMAN_FACE_ID}

# Confidence thresholds for each target class
CONF_THRESHOLDS = {
    BARE_HAND_ID: 0.5,
    HUMAN_FACE_ID: 0.5,
    FOOT_WITHOUT_SHOES_ID: 0.5
}