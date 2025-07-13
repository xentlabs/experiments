
from typing import List, Dict, Any

from shared.types import QualitativeTest


def get_tests() -> List[QualitativeTest]:
    """Returns a list of qualitative xent tests."""
    tests: List[QualitativeTest] = []

    # Highest Xent Token Identification Tests
    highest_xent_texts = [
        "The cat sat on the mat.",
        "The dog barked at the mailman.",
        "I went to the store to buy some bread.",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "The rain in Spain stays mainly in the plain.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "A bird in the hand is worth two in the bush.",
        "The early bird catches the worm.",
    ]

    for text in highest_xent_texts:
        tests.append(
            QualitativeTest(
                test_type="highest_xent_token",
                test_data={"text": text},
            )
        )

    # Comparative Xent Ranking Tests
    comparative_xent_text_groups = [
        [
            "The sun rises in the east.",
            "The platypus is a venomous mammal.",
        ],
        [
            "I am a large language model.",
            "I am a human.",
            "I am a banana.",
        ],
        [
            "The sky is blue.",
            "The sky is green.",
            "The sky is made of cheese.",
        ],
        [
            "The cat sat on the mat.",
            "The cat sat on the television.",
            "The cat sat on the spaceship.",
        ],
    ]

    for text_group in comparative_xent_text_groups:
        tests.append(
            QualitativeTest(
                test_type="comparative_xent",
                test_data={"texts": text_group},
            )
        )

    return tests
