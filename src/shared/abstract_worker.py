import abc
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, TypedDict, TypeVar

# Define generic type variables
ConfigType = TypeVar("ConfigType")
WorkUnitDataType = TypeVar("WorkUnitDataType")
WorkUnitResultType = TypeVar("WorkUnitResultType")


class ResumableTask(abc.ABC, Generic[ConfigType, WorkUnitDataType, WorkUnitResultType]):
    """
    An abstract base class for tasks that can be split into work units,
    executed, and have their results saved and recovered. This allows
    for resuming tasks that might have been interrupted.

    This class is generic over:
    - ConfigType: The type of the initial configuration.
    - WorkUnitDataType: The type of the data for an individual work unit.
    - WorkUnitResultType: The type of the result produced by a work unit.
    """

    def __init__(self, output_dir: str, results_file_extension: str = ".json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results_file_extension = results_file_extension
        if not self._results_file_extension.startswith("."):
            self._results_file_extension = "." + self._results_file_extension

    def _get_result_path(self, work_unit_id: str) -> Path:
        safe_work_unit_id = work_unit_id.replace(os.sep, "_").replace("..", "_")
        return self.output_dir / f"{safe_work_unit_id}{self._results_file_extension}"

    @abc.abstractmethod
    def split_work(
        self, initial_configuration: ConfigType
    ) -> List[Tuple[str, WorkUnitDataType]]:
        """
        Splits the overall task into smaller, manageable work units.

        Args:
            initial_configuration: The configuration needed to define the work.

        Returns:
            A list of tuples: (work_unit_id, work_unit_data).
        """
        pass

    @abc.abstractmethod
    def execute_work_unit(self, work_unit_data: WorkUnitDataType) -> WorkUnitResultType:
        """
        Executes a single work unit.

        Args:
            work_unit_data: The data or parameters for this work unit.

        Returns:
            The result of the work unit's execution.
        """
        pass

    def save_work_unit_result(self, work_unit_id: str, result: WorkUnitResultType):
        """
        Saves the result of a single work unit.
        Default implementation uses JSON. Subclasses might need to override this
        if ResultType is not JSON-serializable or if a different format (e.g., pickle)
        is desired (in which case, also change _results_file_extension).
        """
        result_path = self._get_result_path(work_unit_id)
        try:
            with open(result_path, "w") as f:
                json.dump(
                    result, f, indent=4
                )  # Assumes ResultType is JSON serializable
            logging.info(
                f"INFO: Result for work unit '{work_unit_id}' saved to {result_path}"
            )
        except TypeError as e:
            logging.info(
                f"ERROR: TypeError while saving result for '{work_unit_id}'. "
                f"ResultType might not be JSON serializable. Consider overriding save_work_unit_result. Details: {e}"
            )
            raise
        except Exception as e:
            logging.info(
                f"ERROR: Could not save result for work unit '{work_unit_id}' to {result_path}: {e}"
            )
            raise

    def load_work_unit_result(self, work_unit_id: str) -> Optional[WorkUnitResultType]:
        """
        Loads the result of a previously completed work unit.
        Default implementation uses JSON. Override if needed for different serialization.
        """
        result_path = self._get_result_path(work_unit_id)
        if result_path.exists() and result_path.is_file():
            try:
                logging.info(f"Loading result from {result_path}")
                with open(result_path, "r") as f:
                    # Type casting here might be beneficial if you have complex Pydantic models etc.
                    # but json.load returns Any, so static checkers rely on the method's return type hint.
                    loaded_data = json.load(f)
                return loaded_data  # Assumes loaded_data conforms to ResultType
            except json.JSONDecodeError as e:
                logging.info(
                    f"WARNING: Corrupted result file (JSONDecodeError) for '{work_unit_id}' at {result_path}: {e}"
                )
                return None
            except Exception as e:
                logging.info(
                    f"ERROR: Could not load result for '{work_unit_id}' from {result_path}: {e}"
                )
                return None
        return None

    def is_work_unit_done(self, work_unit_id: str) -> bool:
        return self._get_result_path(work_unit_id).is_file()

    def run_all_work_units(
        self,
        initial_configuration: ConfigType,
        force_rerun_all: bool = False,
        force_rerun_failed_load: bool = False,
    ) -> Dict[str, WorkUnitResultType]:
        logging.info("INFO: Starting task execution...")
        work_units = self.split_work(initial_configuration)
        if not work_units:
            logging.info("INFO: No work units to process.")
            return {}

        all_results_this_run: Dict[str, WorkUnitResultType] = {}
        total_units = len(work_units)
        executed_count = 0
        loaded_count = 0
        failed_execution_count = 0
        skipped_corrupt_count = 0

        logging.info(f"INFO: Total work units to process: {total_units}")

        for i, (work_unit_id, work_unit_data) in enumerate(work_units):
            logging.info(
                f"\nINFO: [{i + 1}/{total_units}] Processing work unit ID: '{work_unit_id}'"
            )

            if not force_rerun_all:
                if self.is_work_unit_done(work_unit_id):
                    logging.info(
                        f"INFO: Result file found for '{work_unit_id}'. Attempting to load."
                    )
                    result = self.load_work_unit_result(work_unit_id)
                    if result is not None:
                        logging.info(
                            f"INFO: Successfully loaded result for '{work_unit_id}'. Skipping execution."
                        )
                        all_results_this_run[work_unit_id] = result
                        loaded_count += 1
                        continue
                    else:
                        if force_rerun_failed_load:
                            logging.info(
                                f"WARNING: Failed to load existing result for '{work_unit_id}'. Re-running as 'force_rerun_failed_load' is True."
                            )
                        else:
                            logging.info(
                                f"WARNING: Failed to load existing result for '{work_unit_id}'. Skipping re-run."
                            )
                            skipped_corrupt_count += 1
                            continue
            elif force_rerun_all:
                logging.info(
                    f"INFO: 'force_rerun_all' is True. Executing '{work_unit_id}' regardless of prior state."
                )

            logging.info(f"INFO: Executing work unit '{work_unit_id}'...")
            try:
                execution_result = self.execute_work_unit(work_unit_data)
                self.save_work_unit_result(work_unit_id, execution_result)
                all_results_this_run[work_unit_id] = execution_result
                executed_count += 1
            except Exception as e:
                failed_execution_count += 1
                logging.info(
                    f"ERROR: Execution of work unit '{work_unit_id}' failed: {e}"
                )
                traceback.print_exc()

        logging.info("\n--- Task Execution Summary ---")
        logging.info(f"Total work units defined: {total_units}")
        logging.info(f"Successfully executed this run: {executed_count}")
        logging.info(f"Successfully loaded from cache: {loaded_count}")
        logging.info(f"Failed during execution this run: {failed_execution_count}")
        if skipped_corrupt_count > 0:
            logging.info(
                f"Skipped due to corrupted existing files (and not forced to rerun): {skipped_corrupt_count}"
            )
        logging.info(
            f"Total results collected in this run: {len(all_results_this_run)}"
        )
        logging.info("--- End of Summary ---")

        return all_results_this_run

    def merge_results(
        self, work_unit_ids: Optional[List[str]] = None
    ) -> List[WorkUnitResultType]:
        collected_results: List[WorkUnitResultType] = []
        # ... (implementation similar, ensure loaded results match ResultType) ...
        ids_to_load: List[str] = []

        if work_unit_ids is not None:
            ids_to_load = work_unit_ids
        else:
            ids_to_load = [
                p.stem for p in self.output_dir.glob(f"*{self._results_file_extension}")
            ]

        logging.info(
            f"INFO: Attempting to merge results for {len(ids_to_load)} work unit ID(s)."
        )
        for unit_id in ids_to_load:
            result = self.load_work_unit_result(unit_id)
            if result is not None:
                collected_results.append(result)
            else:
                logging.info(
                    f"WARNING: Could not load result for work unit ID '{unit_id}' during merge operation."
                )
        logging.info(f"INFO: Successfully merged {len(collected_results)} results.")
        return collected_results

    def get_all_results_as_dict(
        self, initial_configuration: Optional[ConfigType] = None
    ) -> Dict[str, WorkUnitResultType]:
        results_dict: Dict[str, WorkUnitResultType] = {}
        # ... (implementation similar) ...
        ids_to_check: List[str] = []

        if initial_configuration is not None:
            try:
                work_units = self.split_work(initial_configuration)
                ids_to_check = [unit_id for unit_id, _ in work_units]
                logging.info(
                    f"INFO: Using initial configuration to identify {len(ids_to_check)} expected work units."
                )
            except Exception as e:
                logging.info(
                    f"ERROR: Failed to split work from initial_configuration for get_all_results_as_dict: {e}. Will scan directory instead."
                )
                ids_to_check = [
                    p.stem
                    for p in self.output_dir.glob(f"*{self._results_file_extension}")
                ]
        else:
            ids_to_check = [
                p.stem for p in self.output_dir.glob(f"*{self._results_file_extension}")
            ]
            logging.info(
                f"INFO: Scanning directory for result files. Found {len(ids_to_check)} potential result files."
            )

        for unit_id in ids_to_check:
            result = self.load_work_unit_result(unit_id)
            if result is not None:
                results_dict[unit_id] = result
        logging.info(f"INFO: Loaded {len(results_dict)} results into dictionary.")
        return results_dict


# Example: Define more specific types for the benchmark task
class LLMBenchmarkConfig(TypedDict):
    prompts: List[str]
    models: List[str]
    # other global settings


class LLMWorkUnitData(TypedDict):
    prompt_text: str
    model_name: str
    prompt_index: int
    model_index: int
    # other per-unit settings


class LLMResult(TypedDict):
    status: str  # e.g., "success", "error"
    llm_response: Optional[str]
    model_queried: str
    original_prompt_snippet: str
    message: Optional[str]  # For errors
    # other result fields


# Concrete implementation specifying the generic types
class MyTypedLLMBenchmarkTask(
    ResumableTask[LLMBenchmarkConfig, LLMWorkUnitData, LLMResult]
):
    def __init__(self, output_dir: str, api_key: str):
        super().__init__(output_dir=output_dir)
        self.api_key = api_key

    def split_work(
        self, initial_configuration: LLMBenchmarkConfig
    ) -> List[Tuple[str, LLMWorkUnitData]]:
        prompts = initial_configuration["prompts"]
        models = initial_configuration["models"]
        work_units: List[Tuple[str, LLMWorkUnitData]] = []
        for i, prompt_text in enumerate(prompts):
            for j, model_name in enumerate(models):
                work_unit_id = f"prompt{i:03d}_model{j:03d}_{model_name.replace('/', '_').replace(':', '_')}"
                work_unit_data: LLMWorkUnitData = {
                    "prompt_text": prompt_text,
                    "model_name": model_name,
                    "prompt_index": i,
                    "model_index": j,
                }
                work_units.append((work_unit_id, work_unit_data))
        return work_units

    def execute_work_unit(self, work_unit_data: LLMWorkUnitData) -> LLMResult:
        import random
        import time

        prompt = work_unit_data["prompt_text"]
        model = work_unit_data["model_name"]
        logging.info(
            f"  SIMULATING (typed): Querying model '{model}' with prompt '{prompt[:30]}...'"
        )
        time.sleep(random.uniform(0.1, 0.2))

        if random.random() < 0.05:  # Simulate an application-level error
            return {
                "status": "error",
                "llm_response": None,
                "model_queried": model,
                "original_prompt_snippet": prompt[:50],
                "message": f"Simulated API error for {model}",
            }

        response_text = f"Typed LLM response from {model} to '{prompt[:15]}...': Result_{random.randint(100, 999)}"
        return {
            "status": "success",
            "llm_response": response_text,
            "model_queried": model,
            "original_prompt_snippet": prompt[:50],
            "message": None,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    # Configuration for the benchmark using the TypedDict
    typed_benchmark_config: LLMBenchmarkConfig = {
        "prompts": ["What is the capital of Spain?", "Explain black holes simply."],
        "models": ["gpt-3.5-turbo-instruct", "gemini-experimental"],
    }

    typed_results_dir = Path("./my_typed_llm_benchmark_output")
    my_typed_task = MyTypedLLMBenchmarkTask(
        output_dir=str(typed_results_dir), api_key="DUMMY_API_KEY_67890"
    )

    logging.info("\n--- TYPED TASK RUN ---")
    run_results = my_typed_task.run_all_work_units(typed_benchmark_config)

    # Now, `run_results` is Dict[str, LLMResult], and your IDE/type checker knows this.
    for result_id, result_data in run_results.items():
        logging.info(
            f"ID: {result_id}, Status: {result_data['status']}, Response: {result_data.get('llm_response', 'N/A')}"
        )

    all_results_map = my_typed_task.get_all_results_as_dict(
        initial_configuration=typed_benchmark_config
    )
    logging.info(f"\nRetrieved {len(all_results_map)} typed results as a dictionary.")
