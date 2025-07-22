from abc import ABC, abstractmethod

from agentless.repair.repair import construct_topn_file_context
from agentless.util.compress_file import get_skeleton
from agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    get_repo_structure,
    get_simple_repo_structure,
    line_wrap_content,
    show_project_structure,
)

MAX_CONTEXT_LENGTH = 128000


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LLMFL(FL):
    obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path and return at most 10 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```
"""

    obtain_irrelevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of folders that are irrelevant to fixing the problem.
Note that irrelevant folders are those that do not need to be modified and are safe to ignored when trying to solve this problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path.
Remember that any subfolders will be considered as irrelevant if you provide the parent folder.
Please ensure that the provided irrelevant folders do not include any important files needed to fix the problem
The returned folders should be separated by new lines and wrapped with ```
For example:
```
folder1/
folder2/folder3/
folder4/folder5/
```
"""

    file_content_template = """
### File: {file_name} ###
{file_content}
"""
    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""

    obtain_relevant_code_combine_top_n_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited.
The possible location outputs should be either "class", "function" or "line".

### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s) wrapped with ```.
"""

    obtain_relevant_code_combine_top_n_no_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class, method, or function names that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class, method, or function names that need to be edited.
### Examples:
```
full_path1/file1.py
function: my_function1
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method
class: MyClass3

full_path3/file3.py
function: my_function2
```

Return just the location(s) wrapped with ```.
"""
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

    obtain_relevant_functions_and_vars_from_raw_files_prompt = """
Please look through the following GitHub Problem Description and Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 300
        self.model_name = model_name
        self.backend = backend
        self.logger = logger

    def _parse_model_return_lines(self, content: str) -> list[str]:
        """解析模型返回的内容，提取```代码块中的文件名列表"""
        if not content:
            return []
        
        lines = content.strip().split("\n")
        result = []
        in_code_block = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("```"):
                if in_code_block:
                    # 结束代码块
                    break
                else:
                    # 开始代码块
                    in_code_block = True
                    continue
            
            if in_code_block and line:
                # 在代码块内，且不是空行
                result.append(line)
        
        return result

    def _replace_none_with_space(self, message):
        """将message中的所有None值替换为空格"""
        if message is None:
            return " "
        
        if isinstance(message, str):
            # 替换字符串中的 "None" 文本为空格
            return message.replace("None", " ")
        
        if isinstance(message, dict):
            # 递归处理字典
            result = {}
            for key, value in message.items():
                result[key] = self._replace_none_with_space(value)
            return result
        
        if isinstance(message, list):
            # 递归处理列表
            return [self._replace_none_with_space(item) for item in message]
        
        if isinstance(message, tuple):
            # 递归处理元组
            return tuple(self._replace_none_with_space(item) for item in message)
        
        # 对于其他类型，如果是None则返回空格，否则返回原值
        return " " if message is None else message

    def localize_irrelevant(self, top_n=1, mock=False):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        message = self.obtain_irrelevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure=show_project_structure(self.structure).strip(),
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=2048,  # self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        f_files = []
        filtered_files = []

        model_identified_files_folder = self._parse_model_return_lines(raw_output)
        # remove any none folder none files
        model_identified_files_folder = [
            x
            for x in model_identified_files_folder
            if x.endswith("/") or x.endswith(".py")
        ]

        for file_content in files:
            file_name = file_content[0]
            if any([file_name.startswith(x) for x in model_identified_files_folder]):
                filtered_files.append(file_name)
            else:
                f_files.append(file_name)

        self.logger.info(raw_output)

        return (
            f_files,
            {
                "raw_output_files": raw_output,
                "found_files": f_files,
                "filtered_files": filtered_files,
            },
            traj,
        )
    # found_files, details, traj
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        found_files = []
        self.logger.info("Obtaining relevant files for problem statement:")
        message = self.obtain_relevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure=show_project_structure(self.structure).strip(),
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        # 示例用法
        # cleaned_message = self._replace_none_with_space(original_message)
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_files = self._parse_model_return_lines(raw_output)
        print(f"Model found files: {model_found_files}")
        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        # sort based on order of appearance in model_found_files
        # found_files = correct_file_paths(model_found_files, files)
        found_files = model_found_files

        self.logger.info(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def localize_function_from_compressed_files(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
        compress_assign: bool = False,
        total_lines=30,
        prefix_lines=10,
        suffix_lines=10,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(
                code,
                compress_assign=compress_assign,
                total_lines=total_lines,
                prefix_lines=prefix_lines,
                suffix_lines=suffix_lines,
            )
            for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = (
            self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
        )
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names, keep_old_order
        )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_function_from_raw_text(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        raw_file_contents = {fn: code for fn, code in file_contents.items()}
        contents = [
            self.file_content_template.format(file_name=fn, file_content=code)
            for fn, code in raw_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = self.obtain_relevant_functions_and_vars_from_raw_files_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names, keep_old_order
        )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_line_from_coarse_function_locs(
        self,
        file_names,
        coarse_locs,
        context_window: int,
        add_space: bool,
        sticky_scroll: bool,
        no_line_number: bool,
        temperature: float = 0.0,
        num_samples: int = 1,
        mock=False,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = self.obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(coarse_locs) > 1:
            self.logger.info(f"reducing to \n{len(coarse_locs)} files")
            coarse_locs.popitem()
            topn_content, file_loc_intervals = construct_topn_file_context(
                coarse_locs,
                file_names,
                file_contents,
                self.structure,
                context_window=context_window,
                loc_interval=True,
                add_space=add_space,
                sticky_scroll=sticky_scroll,
                no_line_number=no_line_number,
            )
            message = template.format(
                problem_statement=self.problem_statement, file_contents=topn_content
            )

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(
            message, num_samples=num_samples, prompt_cache=num_samples > 1
        )

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names, keep_old_order
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )

    def localize_line_from_raw_text(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        num_samples=1,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        raw_file_contents = {
            fn: line_wrap_content(code) for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_template.format(file_name=fn, file_content=code)
            for fn, code in raw_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names, keep_old_order
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)

        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )
class KernelLLMFL(LLMFL):
    """专门用于Linux内核故障定位的FL类"""
    
    @classmethod
    def create_simple(cls, instance_id, problem_statement, model_name, backend, logger, 
                     repo_path, kernel_subdirs=None, **kwargs):
        """
        创建使用简化文件抽取模式的KernelLLMFL实例
        
        Args:
            instance_id: 实例ID
            problem_statement: 问题描述
            model_name: 模型名称
            backend: 后端
            logger: 日志器
            repo_path: Linux内核仓库本地路径
            kernel_subdirs: 内核子目录列表，None则自动检测
        """
        # 使用简化模式获取结构（这里先传入空结构，稍后会用简化模式覆盖）
        dummy_structure = {}
        
        return cls(
            instance_id=instance_id,
            structure=dummy_structure,
            problem_statement=problem_statement,
            model_name=model_name,
            backend=backend,
            logger=logger,
            kernel_subdirs=kernel_subdirs,
            repo_path=repo_path,
            use_simple_mode=True,
            **kwargs
        )
    
    def __init__(self, instance_id, structure, problem_statement, model_name, backend, logger, kernel_subdirs=None, repo_path=None, use_simple_mode=True, **kwargs):
        super().__init__(instance_id, structure, problem_statement, model_name, backend, logger, **kwargs)
        self.kernel_subdirs = kernel_subdirs or self._get_kernel_subdirs(structure)
        self.repo_path = repo_path  # 新增：本地仓库路径
        self.use_simple_mode = use_simple_mode  # 新增：是否使用简化模式

        # --- 新增调试信息 ---
        print(f"--- DEBUG: KernelLLMFL __init__ set self.kernel_subdirs to: {self.kernel_subdirs} ---")
        print(f"--- DEBUG: KernelLLMFL simple mode: {self.use_simple_mode}, repo_path: {self.repo_path} ---")
    
    def get_simple_structure_for_subdir(self, subdir):
        """为指定子目录获取简化的文件结构"""
        if not self.use_simple_mode or not self.repo_path:
            return self.structure  # 回退到原有结构
        
        try:
            # 使用新的简化文件抽取函数
            # (instance_id: str, repo_path: str, subdirs=None, include_extensions=None)
            subdir_structure = get_simple_repo_structure(
                instance_id=f"{self.instance_id}_{subdir}",
                repo_path=self.repo_path,
                subdirs=[subdir],
                include_extensions=['.c', '.h']  # Linux内核常见文件类型
            )
            return subdir_structure
        except Exception as e:
            self.logger.warning(f"简化模式获取子目录 {subdir} 结构失败: {e}")
            return self.structure  # 回退到原有结构
    
    def _estimate_structure_token_count(self, structure):
        """估算结构的token数量"""
        from agentless.util.api_requests import num_tokens_from_messages
        structure_str = show_project_structure(structure).strip()
        return num_tokens_from_messages(structure_str, self.model_name)
    
    def _split_large_subdir(self, subdir, structure, num_splits=5):
        """将大的子目录分割成多个部分"""
        # 收集该子目录下的所有文件
        subdir_files = []
        for item in structure:
            if isinstance(item, dict) and item.get('type') == 'file':
                if item['name'].startswith(subdir + '/'):
                    subdir_files.append(item)
        
        # 如果文件数量不多，直接返回原结构
        if len(subdir_files) <= num_splits:
            return [structure]
        
        # 将文件分成多个组
        files_per_split = len(subdir_files) // num_splits
        remainder = len(subdir_files) % num_splits
        
        split_structures = []
        start_idx = 0
        
        for i in range(num_splits):
            # 计算当前分组的文件数量
            current_split_size = files_per_split + (1 if i < remainder else 0)
            end_idx = start_idx + current_split_size
            
            # 创建当前分组的结构
            current_files = subdir_files[start_idx:end_idx]
            
            # 构建包含目录和当前文件的结构
            split_structure = []
            
            # 添加目录项
            for item in structure:
                if isinstance(item, dict) and item.get('type') == 'directory':
                    if item['name'] == subdir or item['name'].startswith(subdir + '/'):
                        split_structure.append(item)
            
            # 添加当前分组的文件
            split_structure.extend(current_files)
            
            split_structures.append(split_structure)
            start_idx = end_idx
        
        return split_structures
    
    def _get_kernel_subdirs(self, structure):
        """获取内核的一级子目录"""
        subdirs = []
        for item in structure:
            if item['type'] == 'directory' and '/' not in item['name'].strip('/'):
                subdirs.append(item['name'])
        return subdirs
    
    def _filter_structure_by_subdir(self, structure, subdir):
        """按子目录过滤结构树"""
        filtered = []
        for item in structure:
            if item['name'].startswith(subdir + '/') or item['name'] == subdir:
                filtered.append(item)
        return filtered
    
    # checkhere
    def localize_by_subdirs(self, top_n=10, mock=False):
        """分子目录进行定位，然后合并结果"""
        self.repo_path = "/root/Agentless/linux"
        all_results = []
        all_trajs = []
        print("--- DEBUG: Starting localization by subdirs ---")
        print(f"--- DEBUG: Kernel subdirs: {self.kernel_subdirs} ---")
        for subdir in self.kernel_subdirs:
            self.logger.info(f"正在定位子目录: {subdir}")
            print(f"--- DEBUG: Processing subdir: {subdir} ---")
            
            # 使用简化模式获取该子目录的结构
            subdir_structure = self.get_simple_structure_for_subdir(subdir)
            if subdir == 'drivers':
                print(f"--- DEBUG: drivers subdir detected, splitting into 5 parts ---")
                print(f"--- DEBUG: subdir_structure type: {type(subdir_structure)} ---")
                print(f"--- DEBUG: subdir_structure length: {len(subdir_structure) if hasattr(subdir_structure, '__len__') else 'N/A'} ---")
                
                # 打印前几个结构项目来调试
                if isinstance(subdir_structure, (list, tuple)) and len(subdir_structure) > 0:
                    print(f"--- DEBUG: First few structure items: ---")
                    for i, item in enumerate(subdir_structure[:5]):
                        print(f"    {i}: {item}")
                elif isinstance(subdir_structure, dict):
                    print(f"--- DEBUG: Structure is dict with keys: {list(subdir_structure.keys())[:10]} ---")
                
                # 收集drivers目录下的所有子目录和文件
                drivers_subdirs = []
                drivers_files = []
                all_items = []
                
                # 处理不同的结构格式
                if isinstance(subdir_structure, list):
                    items_to_check = subdir_structure
                elif isinstance(subdir_structure, dict):
                    # 如果是字典，可能需要展开
                    items_to_check = []
                    def extract_items(obj, prefix=""):
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                current_path = f"{prefix}/{key}" if prefix else key
                                if value is None:  # 文件
                                    items_to_check.append({'name': current_path, 'type': 'file'})
                                else:  # 目录
                                    items_to_check.append({'name': current_path, 'type': 'directory'})
                                    if isinstance(value, dict):
                                        extract_items(value, current_path)
                    extract_items(subdir_structure)
                else:
                    items_to_check = []
                
                print(f"--- DEBUG: Total items to check: {len(items_to_check)} ---")
                
                for item in items_to_check:
                    if isinstance(item, dict):
                        item_name = item.get('name', '')
                        item_type = item.get('type', '')
                        
                        all_items.append(item)
                        
                        if item_type == 'directory' and item_name.startswith('drivers/'):
                            # 只保留一级子目录，如 drivers/net, drivers/block 等
                            subdir_parts = item_name.split('/')
                            if len(subdir_parts) == 2:  # drivers/xxx 格式
                                drivers_subdirs.append(item)
                        elif item_type == 'file' and item_name.startswith('drivers/'):
                            drivers_files.append(item)
                
                print(f"--- DEBUG: Total drivers subdirs: {len(drivers_subdirs)}, files: {len(drivers_files)}, all_items: {len(all_items)} ---")
                
                # 如果没有找到符合条件的项目，使用所有项目
                if len(drivers_subdirs) == 0 and len(drivers_files) == 0 and len(all_items) > 0:
                    print(f"--- DEBUG: No drivers items found, using all {len(all_items)} items ---")
                    items_to_split = all_items[:141]  # 限制数量避免过多
                elif len(drivers_subdirs) > len(drivers_files):
                    # 按子目录分割
                    items_to_split = drivers_subdirs
                    print(f"--- DEBUG: Splitting by subdirectories ---")
                else:
                    # 按文件分割
                    items_to_split = drivers_files
                    print(f"--- DEBUG: Splitting by files ---")
                
                print(f"--- DEBUG: Items to split: {len(items_to_split)} ---")
                
                # 如果仍然没有项目，跳过drivers处理
                if len(items_to_split) == 0:
                    print(f"--- DEBUG: No items to split, skipping drivers processing ---")
                    continue
                
                # 将项目平均分成5份
                num_splits = 5
                items_per_split = len(items_to_split) // num_splits
                remainder = len(items_to_split) % num_splits
                
                # 循环处理每个分割
                for i in range(num_splits):
                    print(f"--- DEBUG: Processing drivers part {i+1}/{num_splits} ---")
                    
                    # 计算当前分组的项目数量
                    current_split_size = items_per_split + (1 if i < remainder else 0)
                    start_idx = i * items_per_split + min(i, remainder)
                    end_idx = start_idx + current_split_size
                    
                    # 获取当前分组的项目
                    current_items = items_to_split[start_idx:end_idx]
                    
                    # 构建当前分组的结构
                    split_structure = []
                    
                    # 如果原始结构是字典格式，重建为字典
                    if isinstance(subdir_structure, dict):
                        split_dict = {}
                        # 添加drivers根目录
                        split_dict['drivers'] = {}
                        
                        # 添加当前分组的项目
                        for item in current_items:
                            item_name = item.get('name', '')
                            if item_name.startswith('drivers/'):
                                # 去掉drivers/前缀
                                relative_path = item_name[8:]  # 8 = len('drivers/')
                                if item.get('type') == 'file':
                                    # 处理文件路径
                                    path_parts = relative_path.split('/')
                                    current_node = split_dict['drivers']
                                    for j, part in enumerate(path_parts):
                                        if j == len(path_parts) - 1:
                                            current_node[part] = None  # 文件
                                        else:
                                            if part not in current_node:
                                                current_node[part] = {}
                                            current_node = current_node[part]
                                else:
                                    # 处理目录路径
                                    path_parts = relative_path.split('/')
                                    current_node = split_dict['drivers']
                                    for part in path_parts:
                                        if part not in current_node:
                                            current_node[part] = {}
                                        current_node = current_node[part]
                        
                        split_structure = split_dict
                    else:
                        # 如果原始结构是列表格式，保持列表格式
                        # 添加drivers根目录
                        drivers_root = {'name': 'drivers', 'type': 'directory'}
                        split_structure.append(drivers_root)
                        
                        # 添加当前分组的项目
                        split_structure.extend(current_items)
                        
                        # 如果需要，添加相关的子项目
                        if isinstance(subdir_structure, list):
                            for current_item in current_items:
                                current_item_name = current_item.get('name', '')
                                # 添加该项目下的所有子项目
                                for item in items_to_check:
                                    item_name = item.get('name', '')
                                    if item_name.startswith(current_item_name + '/'):
                                        split_structure.append(item)
                    
                    print(f"--- DEBUG: Part {i+1} has {len(current_items)} main items, total structure complexity: {len(str(split_structure))} chars ---")
                    
                    # 如果当前分组为空，跳过
                    if not current_items:
                        print(f"--- DEBUG: Part {i+1} is empty, skipping ---")
                        continue
                    
                    # 为当前分组创建临时FL实例
                    temp_fl = LLMFL(
                        instance_id=f"{self.instance_id}_drivers_part{i+1}",
                        structure=split_structure,
                        problem_statement=self.problem_statement,
                        model_name=self.model_name,
                        backend=self.backend,
                        logger=self.logger
                    )
                    
                    try:
                        found_files, details, traj = temp_fl.localize(top_n=top_n, mock=mock)
                        print(f"--- DEBUG: Found files in drivers part {i+1}: {found_files} ---")
                        
                        if found_files:
                            # 确保文件路径包含子目录前缀
                            prefixed_files = []
                            for f in found_files:
                                if not f.startswith('drivers/'):
                                    prefixed_files.append(f"drivers/{f}")
                                else:
                                    prefixed_files.append(f)
                            all_results.extend([(f, 'drivers') for f in prefixed_files])
                        
                        all_trajs.append({
                            'subdir': f'drivers_part{i+1}',
                            'traj': traj,
                            'found_files': found_files
                        })
                    except Exception as e:
                        self.logger.warning(f"drivers part {i+1} 定位失败: {e}")
                        continue
                
                # drivers处理完成，跳过后续的正常处理逻辑
                continue

            if not subdir_structure:
                self.logger.warning(f"子目录 {subdir} 结构为空，跳过")
                continue
            # here
            # 检查结构大小，如果太大则分割处理
            try:
                token_count = self._estimate_structure_token_count(subdir_structure)
                self.logger.info(f"子目录 {subdir} 预估token数: {token_count}")
                
                # 如果token数超过阈值（比如20000），则分割处理
                if token_count > 60000:
                    self.logger.info(f"子目录 {subdir} token数过多，分割成5份处理")
                    split_structures = self._split_large_subdir(subdir, subdir_structure, num_splits=5)
                    
                    # 对每个分割的结构进行处理
                    for i, split_structure in enumerate(split_structures):
                        split_subdir = f"{subdir}_part{i+1}"
                        self.logger.info(f"正在处理分割子目录: {split_subdir}")
                        
                        temp_fl = LLMFL(
                            instance_id=f"{self.instance_id}_{split_subdir}",
                            structure=split_structure,
                            problem_statement=self.problem_statement,
                            model_name=self.model_name,
                            backend=self.backend,
                            logger=self.logger
                        )
                        
                        try:
                            found_files, details, traj = temp_fl.localize(top_n=top_n, mock=mock)
                            print(f"--- DEBUG: Found files in {split_subdir}: {found_files} ---")
                            
                            if found_files:
                                # 确保文件路径包含子目录前缀
                                prefixed_files = []
                                for f in found_files:
                                    if not f.startswith(subdir + '/'):
                                        prefixed_files.append(f"{subdir}/{f}")
                                    else:
                                        prefixed_files.append(f)
                                all_results.extend([(f, subdir) for f in prefixed_files])
                            
                            all_trajs.append({
                                'subdir': split_subdir,
                                'traj': traj,
                                'found_files': found_files
                            })
                        except Exception as e:
                            self.logger.warning(f"分割子目录 {split_subdir} 定位失败: {e}")
                            continue
                    
                    continue  # 跳过后面的正常处理逻辑
                    
            except Exception as e:
                self.logger.warning(f"检查子目录 {subdir} token数失败: {e}")
                # 继续使用正常流程
            # here
            print(f"creating temp FL instance for subdir: {subdir}")
            # 为该子目录创建临时FL实例
            temp_fl = LLMFL(
                instance_id=f"{self.instance_id}_{subdir}",
                structure=subdir_structure,
                problem_statement=self.problem_statement,
                model_name=self.model_name,
                backend=self.backend,
                logger=self.logger
            )
            print(f"temp FL instance created for subdir: {subdir}")
            
            # 对该子目录进行定位
            try:
                found_files, details, traj = temp_fl.localize(top_n=top_n, mock=mock)
                print(f"--- DEBUG: Found files in subdir {subdir}: {found_files} ---")
                # print(found_files)
                if found_files:
                    # 确保文件路径包含子目录前缀
                    prefixed_files = []
                    for f in found_files:
                        if not f.startswith(subdir + '/'):
                            prefixed_files.append(f"{subdir}/{f}")
                        else:
                            prefixed_files.append(f)
                    all_results.extend([(f, subdir) for f in prefixed_files])
                
                all_trajs.append({
                    'subdir': subdir,
                    'traj': traj,
                    'found_files': found_files
                })
            except Exception as e:
                self.logger.warning(f"子目录 {subdir} 定位失败: {e}")
               # print(f"--- DEBUG: Exception in subdir {subdir}: {str(e)} ---")
                #print(f"--- DEBUG: subdir_structure type: {type(subdir_structure)}, value: {subdir_structure} ---")
                continue
        
        # 如果没有找到任何文件，直接返回
        if not all_results:
            return [], {'subdir_results': all_trajs}, {'merged_trajs': all_trajs}
        print(f"all_results is{all_results}")
        # 构建包含所有相关文件的新结构，然后对这个新结构进行定位
        final_files = self._localize_from_merged_results(all_results, top_n, mock)
        # final_files=[]
        return final_files, {'subdir_results': all_trajs}, {'merged_trajs': all_trajs}
    
    def _localize_from_merged_results(self, file_results, top_n, mock=False):
        """对所有子目录的结果创建新结构，然后进行定位"""
        if not file_results or mock:
            return [f[0] for f in file_results[:top_n]]
        
        # 构建包含所有相关文件的树状结构
        merged_structure = self._build_file_tree_structure(file_results)
        
        # 创建一个新的LLMFL实例来对合并的结果进行定位
        merged_fl = LLMFL(
            instance_id=f"{self.instance_id}_merged",
            structure=merged_structure,
            problem_statement=self.problem_statement,
            model_name=self.model_name,
            backend=self.backend,
            logger=self.logger
        )
        
        # 对合并后的结构进行定位
        try:
            final_files, details, traj = merged_fl.localize(top_n=top_n, mock=mock)
            self.logger.info(f"合并定位结果: {final_files}")
            return final_files
        except Exception as e:
            self.logger.warning(f"合并定位失败: {e}")
            # 如果失败，回退到直接取前top_n个
            return [f[0] for f in file_results[:top_n]]
    
    def _build_file_tree_structure(self, file_results):
        """构建文件的树状结构，兼容show_project_structure函数"""
        tree = {}
        
        for file_path, subdir in file_results:
            # 分解文件路径
            path_parts = file_path.split('/')
            current_node = tree
            
            # 逐级构建目录结构
            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:
                    # 最后一个是文件名，设置为None（纯文件树结构）
                    current_node[part] = None
                else:
                    # 是目录，如果不存在则创建
                    if part not in current_node:
                        current_node[part] = {}
                    current_node = current_node[part]
        
        return tree

    def localize(self, top_n=10, mock=False):
        """重写定位方法，使用子目录分治策略"""
        return self.localize_by_subdirs(top_n=top_n, mock=mock)