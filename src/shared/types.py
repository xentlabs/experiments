from typing import Any, List, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class XentTest(TypedDict):
    texts: List[str]
    example_texts: List[str]


class XentTextTestResultEntry(TypedDict):
    text: str
    expected: float
    answer: float
    full_conversation: List[ChatMessage]


class XentTextTestResult(TypedDict):
    test: XentTest
    results: List[XentTextTestResultEntry]


class XentTestsForModel(TypedDict):
    model: str
    tests: List[XentTest]


class XentTestForModelResult(TypedDict):
    model: str
    results: List[XentTextTestResult]


class XentTests(TypedDict):
    models: List[str]
    tests: List[XentTest]


class QualitativeTest(TypedDict):
    test_type: str
    test_data: dict


class QualitativeResult(TypedDict):
    test: QualitativeTest
    expected: Any
    answer: Any
    full_conversation: List[ChatMessage]
    is_correct: bool


class QualitativeTestsForModel(TypedDict):
    model: str
    tests: List[QualitativeTest]


class QualitativeTestForModelResult(TypedDict):
    model: str
    results: List[QualitativeResult]


class QualitativeTests(TypedDict):
    models: List[str]
    tests: List[QualitativeTest]
