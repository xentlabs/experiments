from typing import Dict, List, Tuple, TypedDict

# A pair of strings to test. The first string should be true.
Comparison = Tuple[str, str]

# A collection of comparisons, broken down into groups.
# The key should is the group name, the value is the tests in that group
Comparisons = Dict[str, List[Comparison]]


class ModelTruthTest(TypedDict):
    model: str
    comparisons: Comparisons


# Definition of a benchmark to run
class TruthTest(TypedDict):
    comparisons: Comparisons
    models: List[str]


class ComparisonResult(TypedDict):
    comparison: Comparison
    results: Tuple[float, float]


# A dictionary of the results in the same grouping as the comparisons
ModelTruthTestResults = Dict[str, List[ComparisonResult]]


class TruthTestResult(TypedDict):
    test_config: TruthTest
    # A map of model names to their results
    results: Dict[str, ModelTruthTestResults]


class TruthTestSummary(TypedDict):
    model: str
    total: int
    correct: int
    percent: float
    fails: List[ComparisonResult]


# A summary of the results for each model
TruthTestSummaries = Dict[str, TruthTestSummary]
