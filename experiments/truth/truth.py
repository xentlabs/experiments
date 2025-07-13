import json
import logging
import os
import pprint
from typing import Any, List, Tuple

import torch
from comparisons import comparisons
from truthtypes import (
    Comparison,
    Comparisons,
    ModelTruthTest,
    ModelTruthTestResults,
    TruthTest,
    TruthTestResult,
    TruthTestSummaries,
)

from shared.abstract_worker import ResumableTask
from shared.model_manager import log_memory_usage, with_model_and_tokenizer
from shared.model_utils import ALL_BASE_MODELS, xent


def perform_comparison(
    comparison: Comparison, model: Any, tokenizer: Any
) -> Tuple[float, float]:
    score0 = xent(comparison[0], model, tokenizer)
    score1 = xent(comparison[1], model, tokenizer)
    return (score0, score1)


def run_test_for_model(comparisons: Comparisons, model_name: str, device: torch.device):
    model_test_result = {}
    try:
        log_memory_usage("Before loading model", device)

        # Create an inner function that will be decorated
        # The decorator will inject model and tokenizer into this function
        @with_model_and_tokenizer(model_name, device)
        def execute_model_test(comparisons: Comparisons, model=None, tokenizer=None):
            result = {}
            logging.info(f"Starting test for {model_name}")

            for key in comparisons:
                result[key] = []
                logging.info(f"Starting group {key} for model {model_name}")

                for comparison in comparisons[key]:
                    logging.info(
                        f"Performing comparison {comparison} for model {model_name}"
                    )
                    comparison_result = perform_comparison(comparison, model, tokenizer)
                    result[key].append(
                        {"comparison": comparison, "results": comparison_result}
                    )

            logging.info(f"Finished test for {model_name}. Results: {result}")
            return result

        # Call the decorated function
        model_test_result = execute_model_test(comparisons)
        log_memory_usage("AFTER executing test", device)

    except Exception:
        logging.exception(f"Error running test for {model_name}")

    log_memory_usage("AFTER exiting try block", device)
    return model_test_result


def summarize(test_result: TruthTestResult) -> TruthTestSummaries:
    summaries = {}
    for model_name, model_results in test_result["results"].items():
        total = 0
        correct = 0
        fails = []
        for group_name, group_results in model_results.items():
            for result in group_results:
                total += 1
                if result["results"][0] < result["results"][1]:
                    correct += 1
                else:
                    fails.append(result)
        percent = (correct / total) * 100 if total > 0 else 0
        summaries[model_name] = {
            "model": model_name,
            "total": total,
            "correct": correct,
            "percent": percent,
            "fails": fails,
        }
    return summaries


def build_test() -> TruthTest:
    return TruthTest(
        comparisons=comparisons,
        models=[
            "gpt2",
            "gpt2-xl",
            # "meta-llama/Llama-4-Scout-17B-16E",
            # "meta-llama/Llama-4-Maverick-17B-128E",
            # "mistralai/Mistral-Small-24B-Base-2501",
            # "mistralai/Mistral-Small-3.1-24B-Base-2503",
            # "deepseek-ai/DeepSeek-V2-Lite",
            # "deepseek-ai/DeepSeek-V3-Base-Override",
            # "Qwen/Qwen3-1.7B",
            # "Qwen/Qwen3-4B",
            # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            # "Qwen/Qwen2.5-14B",
        ],
    )


def build_full_test() -> TruthTest:
    return TruthTest(comparisons=comparisons, models=ALL_BASE_MODELS)


class TruthTester(ResumableTask[TruthTest, ModelTruthTest, ModelTruthTestResults]):
    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
            )
        )

    def split_work(
        self, initial_configuration: TruthTest
    ) -> List[Tuple[str, ModelTruthTest]]:
        return [
            (
                model_name,
                ModelTruthTest(
                    model=model_name, comparisons=initial_configuration["comparisons"]
                ),
            )
            for model_name in initial_configuration["models"]
        ]

    def execute_work_unit(
        self, work_unit_data: ModelTruthTest
    ) -> ModelTruthTestResults:
        model_name = work_unit_data["model"]
        comparisons = work_unit_data["comparisons"]
        logging.info(f"Running test for {model_name}")
        result = run_test_for_model(comparisons, model_name, self.device)
        return result


def run_truth_test(test_config: TruthTest, output_dir: str):
    truth_tester = TruthTester(output_dir=output_dir)
    results = truth_tester.run_all_work_units(test_config)
    full_results = TruthTestResult(test_config=test_config, results=results)
    summary = summarize(full_results)
    logging.info(f"Summary: {summary}")
    pprint.pprint(summary)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        f.write(json.dumps(summary, indent=4))
    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        f.write(json.dumps(full_results, indent=4))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()
    file_handler = logging.FileHandler("./output/truth/log.txt")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    test = build_test()
    # test = build_full_test()
    run_truth_test(test, output_dir="./output/truth")
