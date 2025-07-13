import json
import logging
import math
import pprint
from typing import Dict, List, Tuple, Union, Any

import torch

from shared.abstract_worker import ResumableTask
from shared.model_manager import log_memory_usage, with_model_and_tokenizer
from shared.model_utils import (
    generate_chat_response,
    remove_think_content,
    xent,
    per_token_xent,
)
from shared.types import (
    QualitativeTest,
    QualitativeTests,
    QualitativeTestsForModel,
    QualitativeTestForModelResult,
    QualitativeResult,
)
from shared.qualitative_xent_tests import get_tests

# System prompt for identifying the token with the highest cross-entropy (xent)
SYSTEM_PROMPT_HIGHEST_XENT = """
Your job is to identify the most surprising word in a given text. The 'most surprising' word is the one that is least expected given the preceding text. When you answer, wrap the single most surprising word in <answer> tags. For example, if the text is 'The cat sat on the mat' and the most surprising word is 'mat', you would write: <answer>mat</answer>. Do not include any other text in your response.
"""

# System prompt for ranking texts by their cross-entropy (xent)
SYSTEM_PROMPT_COMPARATIVE_XENT = """
Your job is to rank a list of texts based on how predictable they are. A more predictable text is one where the words flow naturally and are easy to guess. A less predictable text is more surprising or contains unusual information. You will be given a list of texts, each on a new line. Your answer should be the same list of texts, but reordered from most predictable to least predictable. Wrap your entire answer in <answer> tags. Each text should be on a new line.
"""


def extract_answer_highest_xent(response: str) -> str:
    """Extracts the highest xent token from the model's response."""
    resp = remove_think_content(response)
    start = resp.find("<answer>")
    end = resp.find("</answer>")
    if start == -1 or end == -1:
        logging.info(f"No <answer> tag found in the response: {response}")
        return ""
    return resp[start + len("<answer>") : end].strip()


def extract_answer_comparative_xent(response: str) -> List[str]:
    """Extracts the ranked list of texts from the model's response."""
    resp = remove_think_content(response)
    start = resp.find("<answer>")
    end = resp.find("</answer>")
    if start == -1 or end == -1:
        logging.info(f"No <answer> tag found in the response: {response}")
        return []
    content = resp[start + len("<answer>") : end].strip()
    return [line.strip() for line in content.split('\n') if line.strip()]


def run_qualitative_test_for_model(
    test: QualitativeTest, model, tokenizer
) -> QualitativeResult:
    """Runs a single qualitative xent test for a given model."""
    test_type = test["test_type"]
    test_data = test["test_data"]

    if test_type == "highest_xent_token":
        text = test_data["text"]
        prompt = f"Identify the most surprising word in the following text:\n<text>{text}</text>"
        system_prompt = SYSTEM_PROMPT_HIGHEST_XENT
        per_token_xent_values = per_token_xent(text, model, tokenizer)
        expected_result = max(per_token_xent_values, key=lambda item: item[1])[0]
    elif test_type == "comparative_xent":
        texts = test_data["texts"]
        prompt = "Rank the following texts from most predictable to least predictable:\n" + "\n".join(texts)
        system_prompt = SYSTEM_PROMPT_COMPARATIVE_XENT
        xent_values = {text: xent(text, model, tokenizer) for text in texts}
        expected_result = sorted(texts, key=lambda text: xent_values[text])
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    logging.info(f"Running test: {test}")
    logging.info(f"Prompt:\n{prompt}")

    response, full_conversation = generate_chat_response(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model,
        tokenizer,
        max_new_tokens=2048,
    )
    logging.info(f"Raw response: {response}")

    if test_type == "highest_xent_token":
        answer_result = extract_answer_highest_xent(response)
        is_correct = answer_result == expected_result
    elif test_type == "comparative_xent":
        answer_result = extract_answer_comparative_xent(response)
        is_correct = answer_result == expected_result
    else:
        answer_result = None
        is_correct = False

    return QualitativeResult(
        test=test,
        expected=expected_result,
        answer=answer_result,
        full_conversation=full_conversation,
        is_correct=is_correct,
    )


def run_qualitative_tests_for_model(
    tests: List[QualitativeTest], model_name: str, device: torch.device
) -> List[QualitativeResult]:
    """Loads a model and runs all specified qualitative xent tests against it."""
    try:
        log_memory_usage(f"Before loading model {model_name}", device)

        @with_model_and_tokenizer(model_name, device)
        def execute_model_qualitative_test(
            tests: List[QualitativeTest], model=None, tokenizer=None
        ) -> List[QualitativeResult]:
            results: List[QualitativeResult] = []
            logging.info(f"Starting qualitative xent tests for {model_name}")
            log_memory_usage(f"After loading model {model_name}", device)

            if model is None or tokenizer is None:
                logging.error(
                    f"Model or tokenizer not loaded correctly for {model_name}"
                )
                return []

            for test in tests:
                result = run_qualitative_test_for_model(test, model, tokenizer)
                results.append(result)

            logging.info(f"Finished qualitative xent tests for {model_name}.")
            return results

        model_results = execute_model_qualitative_test(tests)
        log_memory_usage(f"After running tests for {model_name}", device)
        return model_results

    except Exception:
        logging.exception(f"Error running qualitative xent tests for {model_name}")
        log_memory_usage(f"After error during tests for {model_name}", device)
        return []


class QualitativeXentTester(
    ResumableTask[
        QualitativeTests, QualitativeTestsForModel, QualitativeTestForModelResult
    ]
):
    """Manages running qualitative xent tests across multiple models."""

    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Using device: {self.device}")

    def split_work(
        self, initial_configuration: QualitativeTests
    ) -> List[Tuple[str, QualitativeTestsForModel]]:
        return [
            (
                model_name,
                QualitativeTestsForModel(
                    model=model_name, tests=initial_configuration["tests"]
                ),
            )
            for model_name in initial_configuration["models"]
        ]

    def execute_work_unit(
        self, work_unit_data: QualitativeTestsForModel
    ) -> QualitativeTestForModelResult:
        model_name = work_unit_data["model"]
        tests = work_unit_data["tests"]
        logging.info(f"Executing qualitative xent work unit for model: {model_name}")
        results_for_model = run_qualitative_tests_for_model(
            tests, model_name, self.device
        )
        return QualitativeTestForModelResult(model=model_name, results=results_for_model)


def run_qualitative_knowing_the_measure():
    """Main function to set up and run the qualitative xent tests."""
    models_to_test = ["Qwen/Qwen2-0.5B-Instruct"]
    tests = get_tests()

    tester = QualitativeXentTester(output_dir="./output/qualitative_xent")
    test_config = QualitativeTests(models=models_to_test, tests=tests)
    all_results = tester.run_all_work_units(test_config)

    logging.info("--- Qualitative Xent Test Execution Finished ---")
    logging.info("Aggregated Results:")
    pprint.pprint(all_results)
    logging.info("Summarizing results...")
    summary = summarize_results(all_results)
    logging.info("Summary of results:")
    pprint.pprint(summary)
    with open("./output/qualitative_xent/summary.json", "w") as f:
        f.write(json.dumps(summary, indent=4))


def summarize_results(
    results: Dict[str, QualitativeTestForModelResult],
) -> Dict[str, Any]:
    """Summarizes the results of the qualitative xent tests."""
    summary = {}
    for model_name, model_result in results.items():
        model_summary = {
            "highest_xent_token": {
                "correct": 0,
                "incorrect": 0,
                "total": 0,
                "correct_results": [],
                "incorrect_results": [],
            },
            "comparative_xent": {
                "correct": 0,
                "incorrect": 0,
                "total": 0,
                "correct_results": [],
                "incorrect_results": [],
            },
        }
        for result in model_result["results"]:
            test_type = result["test"]["test_type"]
            if test_type not in model_summary:
                continue

            model_summary[test_type]["total"] += 1
            if result["is_correct"]:
                model_summary[test_type]["correct"] += 1
                model_summary[test_type]["correct_results"].append(
                    {
                        "test_data": result["test"]["test_data"],
                        "expected": result["expected"],
                        "answer": result["answer"],
                    }
                )
            else:
                model_summary[test_type]["incorrect"] += 1
                model_summary[test_type]["incorrect_results"].append(
                    {
                        "test_data": result["test"]["test_data"],
                        "expected": result["expected"],
                        "answer": result["answer"],
                    }
                )
        summary[model_name] = model_summary
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    file_handler = logging.FileHandler("./output/qualitative_xent/log.txt")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    run_qualitative_knowing_the_measure()
