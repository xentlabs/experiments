import json
import logging
import math
import pprint
from typing import Dict, List, Tuple

import torch

from shared.abstract_worker import ResumableTask
from shared.model_manager import log_memory_usage, with_model_and_tokenizer
from shared.model_utils import generate_chat_response, remove_think_content, xent
from shared.types import (
    XentTest,
    XentTestForModelResult,
    XentTests,
    XentTestsForModel,
    XentTextTestResult,
    XentTextTestResultEntry,
)
from shared.xent_tests import get_tests

# Define a tolerance for comparing float xent values
XENT_COMPARISON_TOLERANCE = 1e-5


SYSTEM_PROMPT_XENT = """
Your job is to provide information about the total cross-entropy (xent) of a given text. Your answer should be in reference to the model weights you are using.

Your score will be the distance between your answer and the actual xent value. Smaller scores are better.

When you answer a question, wrap the content of the actual answer in <answer> tags. For example, if you wanted to provide the value of XXX, then you would write: <answer>XXX</answer>. The actual answer should be a number value.

If you do not provide the answer in the correct format, then your score will be infinity, which is terrible. Make sure you provide your answer inside of <answer> tags.
"""


def build_example(text: str, model, tokenizer) -> str:
    text_xent = xent(text, model, tokenizer)
    text_xent = round(text_xent * 100) / 100
    return f'Example text "{text}":\n<answer>{text_xent}</answer>\n'


def build_prompt_for_total(
    text: str, example_texts: List[str], model, tokenizer
) -> str:
    if len(example_texts) == 0:
        return f"Provide the xent for the requested text inside of <answer></answer> tags. Text: \n<text>{text}</text>\n"

    examples = [build_example(text, model, tokenizer) for text in example_texts]
    return (
        f"Here are some examples of correct answers:\n"
        f"{'\n'.join(examples)}\n"
        f"Provide the xent for the requested text inside of <answer></answer> tags. Text: \n<text>{text}</text>\n"
    )


def extract_answer(response: str) -> float:
    """Extracts the xent value from the model's response."""
    # First, remove the "thinking" content

    resp = remove_think_content(response)
    start = resp.find("<answer>")
    end = resp.find("</answer>")
    if start == -1 or end == -1:
        logging.info(f"No <answer> tag found in the response: {response}")
        try:
            answer = float(resp)
            logging.info("But the entire output is a valid float, so we can use that")
            return answer
        except ValueError:
            logging.info("The entire output wasn't a float. Returning inf")
        return math.inf

    resp = resp[start + len("<answer>") : end]
    try:
        return float(resp)
    except ValueError:
        raise ValueError(f"Could not convert answer to float: {resp}")


def run_xent_test_for_model(test: XentTest, model, tokenizer) -> XentTextTestResult:
    logging.info(f"Running xent test: {test}")

    results: List[XentTextTestResultEntry] = []
    for text in test["texts"]:
        prompt = build_prompt_for_total(text, test["example_texts"], model, tokenizer)

        logging.info(f"Prompt for text '{text}':\n{prompt}")

        response, full_conversation = generate_chat_response(
            [
                {"role": "system", "content": SYSTEM_PROMPT_XENT},
                {"role": "user", "content": prompt},
            ],
            model,
            tokenizer,
            max_new_tokens=2048,
        )
        logging.info(f"Raw response: {response}")
        expected_result = xent(text, model, tokenizer)
        answer_result = extract_answer(response)

        results.append(
            XentTextTestResultEntry(
                text=text,
                expected=expected_result,
                answer=answer_result,
                full_conversation=full_conversation,
            )
        )

    return XentTextTestResult(test=test, results=results)


def run_xent_tests_for_model(
    tests: List[XentTest], model_name: str, device: torch.device
) -> List[XentTextTestResult]:
    """Loads a model and runs all specified xent tests against it."""
    try:
        log_memory_usage(f"Before loading model {model_name}", device)

        @with_model_and_tokenizer(model_name, device)
        def execute_model_xent_test(
            tests: List[XentTest], model=None, tokenizer=None
        ) -> List[XentTextTestResult]:
            results: List[XentTextTestResult] = []
            logging.info(f"Starting xent tests for {model_name}")
            log_memory_usage(f"After loading model {model_name}", device)

            if model is None or tokenizer is None:
                logging.error(
                    f"Model or tokenizer not loaded correctly for {model_name}"
                )
                return []

            for test in tests:
                logging.info(f"Performing xent test {test} for model {model_name}")
                result = run_xent_test_for_model(test, model, tokenizer)
                results.append(result)

            logging.info(f"Finished xent tests for {model_name}.")
            return results

        model_results = execute_model_xent_test(tests)
        log_memory_usage(f"After running tests for {model_name}", device)
        return model_results

    except Exception:
        logging.exception(f"Error running xent tests for {model_name}")
        log_memory_usage(f"After error during tests for {model_name}", device)
        return []


class XentTester(ResumableTask[XentTests, XentTestsForModel, XentTestForModelResult]):
    """Manages running xent tests across multiple models, potentially resumable."""

    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        # Determine compute device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Correct check for MPS
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Using device: {self.device}")

    def split_work(
        self, initial_configuration: XentTests
    ) -> List[Tuple[str, XentTestsForModel]]:
        """Splits the overall tests into per-model work units."""
        return [
            (
                model_name,
                XentTestsForModel(
                    model=model_name, tests=initial_configuration["tests"]
                ),
            )
            for model_name in initial_configuration["models"]
        ]

    def execute_work_unit(
        self, work_unit_data: XentTestsForModel
    ) -> XentTestForModelResult:
        """Executes all tests for a single model."""
        model_name = work_unit_data["model"]
        tests = work_unit_data["tests"]
        logging.info(f"Executing xent work unit for model: {model_name}")
        results_for_model = run_xent_tests_for_model(tests, model_name, self.device)
        return XentTestForModelResult(model=model_name, results=results_for_model)


def get_xent_tests() -> List[XentTest]:
    # tests = [
    #     XentTest(
    #         texts=["Hello, world!", "This is a test."],
    #         example_texts=["Example text."],
    #     ),
    #     XentTest(texts=["A more complex sentence with punctuation?"], example_texts=[]),
    # ]
    tests = get_tests()
    return tests


def run_knowing_the_measure():
    """Main function to set up and run the xent tests."""
    models_to_test = ["Qwen/Qwen2-0.5B-Instruct"]
    # models_to_test = BIG_MODELS

    tests = get_xent_tests()

    tester = XentTester(output_dir="./output/knowing_the_measure")
    test_config = XentTests(models=models_to_test, tests=tests)
    all_results = tester.run_all_work_units(test_config)

    logging.info("--- Xent Test Execution Finished ---")
    logging.info("Aggregated Results:")
    pprint.pprint(all_results)
    logging.info("Summarizing results...")
    summary = summarize_results(all_results)
    logging.info("Summary of results:")
    pprint.pprint(summary)
    # Write the summary to a file
    with open("./output/knowing_the_measure/xent_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=4))


def summarize_results(
    results: Dict[str, XentTestForModelResult],
):
    """Summarizes the results of the xent tests."""
    summary = {}
    for model_name, model_result in results.items():
        summary[model_name] = {
            "sum": 0,
            "sum_with_examples": 0,
            "sum_without_examples": 0,
            "with_examples": [],
            "without_examples": [],
            "fails": [],
            "fails_with_examples": [],
            "fails_without_examples": [],
        }
        for test_result in model_result["results"]:
            test = test_result["test"]
            for entry in test_result["results"]:
                if entry["answer"] == math.inf:
                    summary[model_name]["fails"].append({"text": entry["text"]})
                    if len(test["example_texts"]) > 0:
                        summary[model_name]["fails_with_examples"].append(
                            {"text": entry["text"]}
                        )
                    else:
                        summary[model_name]["fails_without_examples"].append(
                            {"text": entry["text"]}
                        )
                    continue

                score = abs(entry["expected"] - entry["answer"])
                summary[model_name]["sum"] += score
                if len(test["example_texts"]) > 0:
                    summary[model_name]["sum_with_examples"] += score
                    summary[model_name]["with_examples"].append(
                        {
                            "text": entry["text"],
                            "expected": entry["expected"],
                            "answer": entry["answer"],
                        }
                    )
                else:
                    summary[model_name]["sum_without_examples"] += score
                    summary[model_name]["without_examples"].append(
                        {
                            "text": entry["text"],
                            "expected": entry["expected"],
                            "answer": entry["answer"],
                        }
                    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    file_handler = logging.FileHandler("./output/knowing_the_measure/log.txt")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    run_knowing_the_measure()
