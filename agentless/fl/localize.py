import argparse
import concurrent.futures
import json
import os
from threading import Lock

from datasets import load_dataset
from tqdm import tqdm

from agentless.fl.FL import LLMFL, KernelLLMFL
from agentless.util.preprocess_data import (
    check_contains_valid_loc,
    filter_none_python,
    filter_out_test_files,
    get_repo_structure,
)
from agentless.util.utils import load_existing_instance_ids, load_jsonl, setup_logger

MAX_RETRIES = 5

def load_kernel_dataset(jsonl_path):
    """加载Linux内核数据集"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 转换为FL需要的格式
            processed_item = {
                'instance_id': item['id'],
                'commit':item['commit'],
                'problem_statement': f"Bug Title: {item.get('bug_title', 'Unknown')}\n\nReport: {item['report']}"
            }
            data.append(processed_item)
    return data

def localize_irrelevant_instance(
    bug, args, swe_bench_data, existing_instance_ids, write_lock=None
):
    instance_id = bug["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if bug["instance_id"] in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
        return

    logger.info(f"================ localize {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = get_repo_structure(
        instance_id, bug["repo"], bug["base_commit"], "playground"
    )

    # filter_none_python(structure)  # some basic filtering steps
    filter_out_test_files(structure)

    found_files = []
    found_related_locs = {}
    found_edit_locs = {}

    additional_artifact_loc_file = None
    additional_artifact_loc_related = None
    additional_artifact_loc_edit_location = None
    file_traj, related_loc_traj, edit_loc_traj = {}, {}, {}

    # file level localization
    if args.file_level:
        fl = LLMFL(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
        )
        found_files, additional_artifact_loc_file, file_traj = fl.localize_irrelevant(
            mock=args.mock
        )
    else:
        raise NotImplementedError

    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "found_files": found_files,
                    "additional_artifact_loc_file": additional_artifact_loc_file,
                    "file_traj": file_traj,
                    "found_related_locs": found_related_locs,
                    "additional_artifact_loc_related": additional_artifact_loc_related,
                    "related_loc_traj": related_loc_traj,
                    "found_edit_locs": found_edit_locs,
                    "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                    "edit_loc_traj": edit_loc_traj,
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()


def localize_instance(
    bug, args, swe_bench_data, start_file_locs, existing_instance_ids, write_lock=None
):
    instance_id = bug["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
        return

    structure = get_repo_structure(
        instance_id, bug["repo"], bug["base_commit"], "playground"
    )

    logger.info(f"================ localize {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]

    # filter_none_python(structure)  # some basic filtering steps
    filter_out_test_files(structure)

    found_files = []
    found_related_locs = {}
    found_edit_locs = {}
    additional_artifact_loc_file = None
    additional_artifact_loc_related = None
    additional_artifact_loc_edit_location = None
    file_traj, related_loc_trajs, edit_loc_traj = {}, [], {}

    # file level localization
    if args.file_level:
        fl = LLMFL(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
        )
        found_files, additional_artifact_loc_file, file_traj = fl.localize(
            mock=args.mock
        )
    else:
        # assume start_file is provided
        for locs in start_file_locs:
            if locs["instance_id"] == instance_id:
                found_files = locs["found_files"]
                if "additional_artifact_loc_file" in locs:
                    additional_artifact_loc_file = locs["additional_artifact_loc_file"]
                    file_traj = locs["file_traj"]
                if "found_related_locs" in locs:
                    found_related_locs = locs["found_related_locs"]
                    additional_artifact_loc_related = locs[
                        "additional_artifact_loc_related"
                    ]
                    related_loc_trajs = locs["related_loc_traj"]
                break

        if len(found_files) == 0:
            return

    # related class, functions, global var localization
    if args.related_level:
        if len(found_files) != 0:
            trying_temp = 0  # related always try with temp 0
            related_loc_trajs = []
            for _ in range(MAX_RETRIES):
                pred_files = found_files[: args.top_n]
                fl = LLMFL(
                    instance_id,
                    structure,
                    problem_statement,
                    args.model,
                    args.backend,
                    logger,
                )
                additional_artifact_loc_related = []
                found_related_locs = {}
                related_loc_traj = {}
                if args.compress and not args.related_level_separate_file:
                    (
                        found_related_locs,
                        additional_artifact_loc_related,
                        related_loc_traj,
                    ) = fl.localize_function_from_compressed_files(
                        pred_files,
                        mock=args.mock,
                        temperature=trying_temp,
                        keep_old_order=args.keep_old_order,
                        compress_assign=args.compress_assign,
                        total_lines=args.compress_assign_total_lines,
                        prefix_lines=args.compress_assign_prefix_lines,
                        suffix_lines=args.compress_assign_suffix_lines,
                    )
                    additional_artifact_loc_related = [additional_artifact_loc_related]
                    related_loc_trajs.append(related_loc_traj)

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break

                    logger.info(
                        f"No valid related locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs

                elif args.compress and args.related_level_separate_file:
                    additional_artifact_loc_related = []
                    found_related_locs = {fn: [] for fn in pred_files}
                    related_loc_traj = []
                    for i, pred_file in enumerate(pred_files):
                        (
                            found_related_locs_i,
                            additional_artifact_loc_related_i,
                            related_loc_traj_i,
                        ) = fl.localize_function_from_compressed_files(
                            [pred_file],
                            mock=args.mock,
                            temperature=trying_temp,
                            compress_assign=args.compress_assign,
                            total_lines=args.compress_assign_total_lines,
                            prefix_lines=args.compress_assign_prefix_lines,
                            suffix_lines=args.compress_assign_suffix_lines,
                        )
                        found_related_locs[pred_file] = found_related_locs_i[pred_file]
                        additional_artifact_loc_related.append(
                            additional_artifact_loc_related_i
                        )
                        related_loc_traj.append(related_loc_traj_i)
                    related_loc_trajs.append(related_loc_traj)

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs
                else:
                    # directly use raw code file instead of skeleton format as ablation
                    (
                        found_related_locs,
                        additional_artifact_loc_related,
                        related_loc_traj,
                    ) = fl.localize_function_from_raw_text(
                        pred_files,
                        mock=args.mock,
                        temperature=trying_temp,
                        keep_old_order=args.keep_old_order,
                    )
                    additional_artifact_loc_related = [additional_artifact_loc_related]
                    related_loc_trajs.append(related_loc_traj)

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break

                    logger.info(
                        f"No valid related locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs

    if args.fine_grain_line_level:
        if len(found_files) != 0:
            # Only supports the following args for now
            trying_temp = args.temperature
            for _ in range(MAX_RETRIES):
                pred_files = found_files[: args.top_n]
                fl = LLMFL(
                    instance_id,
                    structure,
                    problem_statement,
                    args.model,
                    args.backend,
                    logger,
                )
                if not args.direct_edit_loc:
                    coarse_found_locs = found_related_locs
                    (
                        found_edit_locs,
                        additional_artifact_loc_edit_location,
                        edit_loc_traj,
                    ) = fl.localize_line_from_coarse_function_locs(
                        pred_files,
                        coarse_found_locs,
                        context_window=args.context_window,
                        add_space=args.add_space,
                        no_line_number=args.no_line_number,
                        sticky_scroll=args.sticky_scroll,
                        mock=args.mock,
                        temperature=trying_temp,
                        num_samples=args.num_samples,
                        keep_old_order=args.keep_old_order,
                    )
                    additional_artifact_loc_edit_location = [
                        additional_artifact_loc_edit_location
                    ]

                    sample_valid = False
                    if args.num_samples > 1:
                        for found_edit_loc in found_edit_locs:
                            if check_contains_valid_loc(
                                found_edit_loc, structure=structure
                            ):
                                # at least one set of location contains valid edit locs is okay
                                sample_valid = True
                                break
                    else:
                        if check_contains_valid_loc(
                            found_edit_locs, structure=structure
                        ):
                            break

                    if sample_valid:
                        break

                    logger.info(
                        f"No valid edit locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs
                else:
                    # directly use the raw source code to get to edit location, used as ablation setting
                    (
                        found_edit_locs,
                        additional_artifact_loc_edit_location,
                        edit_loc_traj,
                    ) = fl.localize_line_from_raw_text(
                        pred_files,
                        mock=args.mock,
                        temperature=trying_temp,
                        num_samples=args.num_samples,
                        keep_old_order=args.keep_old_order,
                    )
                    additional_artifact_loc_edit_location = [
                        additional_artifact_loc_edit_location
                    ]

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break

                    sample_valid = False
                    if args.num_samples > 1:
                        for found_edit_loc in found_edit_locs:
                            if check_contains_valid_loc(
                                found_edit_loc, structure=structure
                            ):
                                # at least one set of location contains valid edit locs is okay
                                sample_valid = True
                                break
                    else:
                        if check_contains_valid_loc(
                            found_edit_locs, structure=structure
                        ):
                            break

                    if sample_valid:
                        break

                    logger.info(
                        f"No valid edit locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs

    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "found_files": found_files,
                    "additional_artifact_loc_file": additional_artifact_loc_file,
                    "file_traj": file_traj,
                    "found_related_locs": found_related_locs,
                    "additional_artifact_loc_related": additional_artifact_loc_related,
                    "related_loc_traj": related_loc_trajs,
                    "found_edit_locs": found_edit_locs,
                    "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                    "edit_loc_traj": edit_loc_traj,
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()


def localize_irrelevant(args):
    swe_bench_data = load_dataset(args.dataset, split="test")
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )
    if args.num_threads == 1:
        for bug in tqdm(swe_bench_data, colour="MAGENTA"):
            localize_irrelevant_instance(
                bug, args, swe_bench_data, existing_instance_ids
            )
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    localize_irrelevant_instance,
                    bug,
                    args,
                    swe_bench_data,
                    existing_instance_ids,
                    write_lock,
                )
                for bug in swe_bench_data
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(swe_bench_data),
                colour="MAGENTA",
            ):
                future.result()


def localize(args):
    swe_bench_data = load_dataset(args.dataset, split="test")
    start_file_locs = load_jsonl(args.start_file) if args.start_file else None
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    if args.num_threads == 1:
        for bug in tqdm(swe_bench_data, colour="MAGENTA"):
            localize_instance(
                bug, args, swe_bench_data, start_file_locs, existing_instance_ids
            )
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    localize_instance,
                    bug,
                    args,
                    swe_bench_data,
                    start_file_locs,
                    existing_instance_ids,
                    write_lock,
                )
                for bug in swe_bench_data
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(swe_bench_data),
                colour="MAGENTA",
            ):
                future.result()


def merge(args):
    """Merge predicted locations."""
    start_file_locs = load_jsonl(args.start_file)

    def merge_locs(sample_found_locs: list[dict]):
        merged_found_locs = {}
        for locs in sample_found_locs:
            for fn, file_found_locs in locs.items():
                if isinstance(file_found_locs, str) and file_found_locs.strip():
                    merged_found_locs.setdefault(fn, [""])[0] += "\n" + file_found_locs
                elif "\n".join(file_found_locs).strip():
                    merged_found_locs.setdefault(fn, [""])[0] += "\n" + "\n".join(
                        file_found_locs
                    )
        return merged_found_locs

    # Dump each location sample.
    for st_id in range(args.num_samples):
        en_id = st_id
        merged_locs = []
        for locs in start_file_locs:
            merged_found_locs = []
            if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
                merged_found_locs = merge_locs(
                    locs["found_edit_locs"][st_id : st_id + 1]
                )
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
        with open(
            f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")


def check_valid_args(args):
    assert (
        not os.path.exists(args.output_file) or args.skip_existing
    ), "Output file already exists and not set to skip existing localizations"

    assert not (
        args.file_level and args.start_file
    ), "Cannot use both file_level and start_file"

    assert not (
        args.file_level and args.fine_grain_line_level and not args.related_level
    ), "Cannot use both file_level and fine_grain_line_level without related_level"

    assert not (
        (not args.file_level) and (not args.start_file)
    ), "Must use either file_level or start_file"

    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

# 2. 新增专门用于执行分层定位的函数
from agentless.util.preprocess_data import get_repo_structure

def localize_hierarchical_instance(
    bug, args, swe_bench_data, start_file_locs, existing_instance_ids, write_lock=None
):
    instance_id = bug["instance_id"]
    # print(instance_id)
    log_file = os.path.join(args.output_folder, "localization_logs", f"{instance_id}.log")
    # print(bug)
    # print(bug["commit"])
    if args.target_id and args.target_id != instance_id:
        return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id} with DIRECT HIERARCHICAL strategy")

    if instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {instance_id}")
        return

    problem_statement = bug["problem_statement"]
    
    # 获取代码结构
    structure = get_repo_structure(
        instance_id, 
        "torvalds/linux", 
        bug["commit"], 
        "playground",
        subdirs=args.target_subdirectories,
        simple_mode=True,  # 使用简单模式以避免过多的细节
    )
    # print(structure)
    print(f"INFO: Retrieved structure for {instance_id} with subdirs {args.target_subdirectories}")
    if not structure:
        logger.error(f"Failed to get repo structure for {instance_id}. Skipping.")
        return
    print(f"INFO: Structure for {instance_id} retrieved successfully.")

    
    # 关键步骤：实例化 KernelLLMFL
    kernel_fl = KernelLLMFL(
        instance_id=instance_id,
        structure=structure,
        problem_statement=problem_statement,
        model_name=args.model,
        backend=args.backend,
        logger=logger,
        # 传入由命令行指定的目标子目录
        kernel_subdirs=args.target_subdirectories
    )
    
    print(f"INFO: Starting hierarchical localization for {instance_id} with subdirs {args.target_subdirectories}")
    # 调用 localize，这将自动触发分治和重排逻辑
    try:
        print(f"INFO: Localizing {instance_id} with top_n={args.top_n} and mock={args.mock}")
        final_ranked_files, details, traj = kernel_fl.localize(top_n=args.top_n, mock=args.mock)
        print(f"INFO: Hierarchical localization completed for {instance_id}")
    except Exception as e:
        logger.error(f"Hierarchical localization failed for {instance_id}: {e}")
        return

    # 保存分层定位的详细结果
    if write_lock:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        # details 变量包含了我们之前定义的 subdir_results
        # traj 变量包含了 merged_trajs
        output_data = details 
        output_data["instance_id"] = instance_id
        output_data["strategy"] = "direct_hierarchical"
        output_data["final_ranked_files"] = final_ranked_files
        
        f.write(json.dumps(output_data) + "\n")
    if write_lock:
        write_lock.release()

def dispatch_localization(args):
    # 根据数据集来源加载数据
    if "SWE-bench" in args.dataset:
        dataset = load_dataset(args.dataset, split="test")
        # 将 problem_statement 格式化后存入一个字典，以便快速查找
        problem_statement_map = {item["instance_id"]: item["problem_statement"] for item in dataset}
        full_dataset = dataset
    else: 
        full_dataset = load_kernel_dataset(args.dataset)
        problem_statement_map = {item["instance_id"]: item["problem_statement"] for item in full_dataset}

    start_file_locs = load_jsonl(args.start_file) if args.start_file else None
    existing_instance_ids = load_existing_instance_ids(args.output_file) if args.skip_existing else set()

    # 根据 --hierarchical 标志选择要执行的目标函数
    if args.hierarchical:
        target_func = localize_hierarchical_instance
        print("INFO: Running in HIERARCHICAL localization mode.")
    else:
        target_func = localize_instance
        print("INFO: Running in STANDARD localization mode.")
    
    # `bug` 对象现在应该包含 instance_id, repo, base_commit
    # `swe_bench_data` 参数现在传递 problem_statement_map
    if args.num_threads == 1:
        for bug in tqdm(full_dataset, colour="CYAN"):
            target_func(bug, args, problem_statement_map, start_file_locs, existing_instance_ids)
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(
                    target_func,
                    bug, args, problem_statement_map, start_file_locs, existing_instance_ids, write_lock
                )
                for bug in full_dataset
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(full_dataset), colour="CYAN"):
                future.result()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchical", action="store_true", help="Use the hierarchical localization strategy for Kernels.")
    parser.add_argument(
        "--target_subdirectories",
        nargs='+',
        default=['fs', 'drivers', 'kernel', 'net', 'arch', 'mm', 'include'],
        help="A list of top-level subdirectories to scan in hierarchical mode. Example: --target_subdirectories fs net"
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--start_file",
        type=str,
        help="""previous output file to start with to reduce
        the work, should use in combination without --file_level""",
    )
    parser.add_argument("--file_level", action="store_true")
    parser.add_argument("--related_level", action="store_true")
    parser.add_argument("--fine_grain_line_level", action="store_true")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--compress_assign", action="store_true")
    parser.add_argument("--compress_assign_total_lines", type=int, default=30)
    parser.add_argument("--compress_assign_prefix_lines", type=int, default=10)
    parser.add_argument("--compress_assign_suffix_lines", type=int, default=10)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--related_level_separate_file", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--keep_old_order", action="store_true")
    parser.add_argument("--irrelevant", action="store_true")
    parser.add_argument("--direct_edit_loc", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-coder",
        choices=[
            "gpt-4o-2024-05-13",
            "deepseek-coder",
            "gpt-4o-mini-2024-07-18",
            "claude-3-5-sonnet-20241022",
        ],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="deepseek",
        choices=["openai", "deepseek", "anthropic"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
    #    choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
        help="Current supported dataset for evaluation",
    )

    args = parser.parse_args()
    args.output_file = os.path.join(args.output_folder, args.output_file)
    check_valid_args(args)

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.merge:
        merge(args)
    elif args.irrelevant:
        localize_irrelevant(args)
    else:
        dispatch_localization(args)


if __name__ == "__main__":
    main()
