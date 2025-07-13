from typing import List, TypedDict


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
