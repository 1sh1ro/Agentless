import csv
import json
import os
import subprocess
import requests
import ast
import time
import glob
from clang.cindex import Index, CursorKind

# --- 1. 配置区域 ---

# 请将此路径修改为您本地的 Linux 内核代码仓库路径
LINUX_REPO_PATH = "/root/Agentless/linux/"

# 输入文件
CSV_FILE_PATH = 'merged_output.csv'  # 您的 CSV 文件名
JSONL_FILE_PATH = '2.jsonl'

# 大模型 API 配置 (请填入您的真实信息)
API_URL = "https://api.deepseek.com/chat/completions"
API_KEY = "" # use your own key

# 输出目录
OUTPUT_DIR = "llm_results"

# 最大上下文长度（与原始版本保持一致）
MAX_CONTEXT_LENGTH = 60000

# 文件内容模板（与原始版本保持一致）
FILE_CONTENT_TEMPLATE = """
### File: {file_name} ###
{file_content}
"""

# --- 2. Prompt 模板 ---

OBTAIN_RELEVANT_FUNCTIONS_PROMPT = """
Please analyze the following GitHub Problem Description and Structured Code Information.
The code information has been extracted using libclang and shows functions, structures, variables, and macros.

Your task is to identify the Top-10 most relevant code locations that need inspection or editing to fix the problem.
Focus on the actual functions, structures, variables, and macros that are directly related to the issue.

### GitHub Problem Description ###
{problem_statement}

### Structured Code Information ###
{file_contents}

###

Please provide the complete set of locations in this format:
- For functions: function: function_name
- For structures: struct: struct_name
- For variables: variable: variable_name
- For macros: macro: macro_name

### Examples:
    fs/ext4/inode.c
function: ext4_write_begin
function: ext4_write_end
struct: ext4_inode_info

net/core/skbuff.c
function: skb_copy_data
variable: skb_head_cache
macro: SKB_DATA_ALIGN

mm/page_alloc.c
function: alloc_pages
function: free_pages
struct: free_area
Return just the locations wrapped with ```. Focus on the most relevant items based on the problem description.
"""


# --- 3. 辅助函数 ---

def load_dataset_to_dict(jsonl_path):
    """一次性加载 JSONL 数据到字典中，以 ID 为键，方便快速查找。"""
    dataset = {}
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        dataset[data['id']] = data
                except json.JSONDecodeError:
                    print(f"警告：跳过格式错误的 JSON 行: {line.strip()}")
        print(f"成功加载 {len(dataset)} 条数据从 {jsonl_path}")
        return dataset
    except FileNotFoundError:
        print(f"错误：找不到数据集文件 {jsonl_path}")
        exit(1)

def cleanup_git_locks(repo_path):
    """清理Git仓库中的所有锁文件"""
    print(f"🧹 开始清理Git锁文件...")
    lock_patterns = [
        '.git/index.lock', '.git/HEAD.lock', '.git/config.lock',
        '.git/refs/heads/*.lock', '.git/refs/remotes/*/*.lock', '.git/objects/pack/*.lock',
    ]
    cleaned_count = 0
    for pattern in lock_patterns:
        full_pattern = os.path.join(repo_path, pattern)
        for lock_file in glob.glob(full_pattern):
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    print(f"   ✅ 删除: {os.path.relpath(lock_file, repo_path)}")
                    cleaned_count += 1
            except Exception as e:
                print(f"   ❌ 删除失败 {lock_file}: {e}")
    if cleaned_count > 0:
        print(f"🎯 共清理了 {cleaned_count} 个锁文件")
    else:
        print(f"ℹ️  没有发现锁文件")
    return cleaned_count > 0

def checkout_commit(repo_path, commit_hash):
    """在指定的仓库路径中切换到特定的 commit。"""
    if not os.path.isdir(repo_path):
        print(f"错误：指定的 Linux 仓库路径不存在: {repo_path}")
        return False
    print(f"正在切换到 commit: {commit_hash} ...")
    try:
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, check=True, capture_output=True, text=True)
        result = subprocess.run(['git', 'checkout', commit_hash], cwd=repo_path, check=True, capture_output=True, text=True)
        print("✅ 切换 commit 成功。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 切换到 commit {commit_hash} 失败。")
        print(f"Git 命令输出:\n{e.stderr}")
        print("🔄 正在尝试清理锁文件后重试...")
        cleanup_git_locks(repo_path)
        try:
            result = subprocess.run(['git', 'checkout', commit_hash], cwd=repo_path, check=True, capture_output=True, text=True)
            print("✅ 清理后切换成功。")
            return True
        except Exception as e2:
             print(f"❌ 清理后重试依然失败: {e2}")
             return False
    except FileNotFoundError:
        print("❌ 'git' 命令未找到。请确保 Git 已安装并位于您的 PATH 中。")
        return False

def read_file_content(file_path):
    """使用libclang解析文件内容，提取结构化信息"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return f"# 文件不存在: {file_path}\n"
    print(f"📂 开始处理文件: {os.path.basename(file_path)}")
    try:
        if file_path.endswith(('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp')):
            return extract_c_structures(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                return FILE_CONTENT_TEMPLATE.format(file_name=file_path, file_content=''.join(lines))
    except Exception as e:
        print(f"❌ 文件处理失败: {e}")
        return f"# 读取文件失败: {file_path}, 错误: {e}\n"

def extract_c_structures(file_path):
    """使用libclang提取C/C++文件的结构化信息"""
    try:
        index = Index.create()
        translation_unit = index.parse(file_path, args=['-I' + os.path.join(LINUX_REPO_PATH, 'include')])

        diagnostics = list(translation_unit.diagnostics)
        if diagnostics:
            # 只显示警告和错误，忽略提示信息
            severe_diags = [d for d in diagnostics if d.severity >= 3]
            if severe_diags:
                print(f"⚠️  libclang发现 {len(severe_diags)} 个严重诊断信息 (仅显示前3个):")
                for diag in severe_diags[:3]:
                    print(f"   - {diag.severity}: {diag.spelling} at {diag.location}")

        structures = [f"### File: {file_path} ###"]

        # 我们只提取顶层的函数定义，以简化输出
        for node in translation_unit.cursor.get_children():
            if node.location.file and os.path.samefile(node.location.file.name, file_path):
                 if node.kind == CursorKind.FUNCTION_DECL and node.is_definition():
                    func_name = node.spelling
                    if func_name:
                        structures.append(f"function: {func_name}")

        if len(structures) <= 1:
             print(f"⚠️  libclang未能提取任何函数，回退到文本模式")
             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                 return f.read()

        return '\n'.join(structures) + '\n'
    except Exception as e:
        print(f"❌ libclang解析失败: {e}。回退到文本读取模式...")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as read_error:
            print(f"❌ 文件读取也失败: {read_error}")
            return f"# 读取文件失败: {file_path}, 错误: {e}\n"


def query_llm(prompt):
    """
    向大模型发送请求。
    如果成功，返回结果字符串。
    如果遇到token超限错误，返回特殊标识 "TOKEN_LIMIT_EXCEEDED"。
    如果遇到其他错误，返回 None。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in code analysis and debugging."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": False
    }
    print("正在向大模型发送请求...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180)

        # 检查是否是API错误
        if response.status_code != 200:
            print(f"API请求失败，状态码: {response.status_code}")
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "").lower()
            error_code = error_data.get("error", {}).get("code", "")

            # ** 关键改动：专门识别token超限错误 **
            if "context_length_exceeded" in error_code or "token limits" in error_message:
                print("识别到Token超限错误。")
                return "TOKEN_LIMIT_EXCEEDED"

            print(f"响应内容: {response.text}")
            return None

        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            result_text = response_data["choices"][0]["message"].get("content", "")
            print("✅ 成功从LLM获取响应。")
            return result_text.strip()
        else:
            print(f"API响应格式异常: {response_data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"错误：API 请求失败: {e}")
        return None

def num_tokens_from_messages(message, model_name):
    """估算token数量的简单实现"""
    return len(message) // 4

def extract_code_blocks(text):
    """从LLM输出中提取代码块"""
    if '```' in text:
        parts = text.split('```')
        if len(parts) >= 3:
            return parts[1].strip()
    return text.strip()

def extract_locs_for_files(model_found_locs, file_names):
    """解析定位结果并按文件分组"""
    result = {}
    lines = model_found_locs.split('\n')
    current_file = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 检查是否是文件路径
        # 更稳健的检查，确保line是已知的相关文件之一
        is_file_path = False
        for fname in file_names:
            if fname in line:
                 is_file_path = True
                 current_file = fname
                 break

        if is_file_path:
             if current_file not in result:
                result[current_file] = []
        elif current_file and line.startswith(('function:', 'struct:', 'variable:', 'macro:')):
            if current_file in result:
                result[current_file].append(line)
    return result

# --- 4. 主执行逻辑 ---

def main():
    """主函数，协调整个流程。"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    if os.path.exists(LINUX_REPO_PATH):
        cleanup_git_locks(LINUX_REPO_PATH)
    else:
        print(f"⚠️  Linux仓库路径不存在: {LINUX_REPO_PATH}")
        return

    dataset = load_dataset_to_dict(JSONL_FILE_PATH)
    if not dataset:
        return

    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            all_rows = list(reader)
            total_rows = len(all_rows)
            start_index = 0

            print(f"开始处理 CSV 文件: {CSV_FILE_PATH}")
            print(f"总共 {total_rows} 行数据，从第 {start_index + 1} 行开始处理")
            print("-" * 50)

            for i in range(start_index, total_rows):
                row = all_rows[i]
                instance_id, files_str = row
                print(f"\n处理第 {i+1} 行: instance_id = {instance_id} (进度: {i+1-start_index}/{total_rows-start_index})")

                if instance_id not in dataset:
                    print(f"警告：在 {JSONL_FILE_PATH} 中找不到 instance_id '{instance_id}'，跳过此行。")
                    continue

                bug_data = dataset[instance_id]
                commit_hash, problem_statement = bug_data.get('commit'), bug_data.get('report')
                if not commit_hash or not problem_statement:
                    print(f"警告：instance_id '{instance_id}' 的数据不完整，跳过。")
                    continue

                if not checkout_commit(LINUX_REPO_PATH, commit_hash):
                    print(f"❌ 无法切换到 commit，跳过 instance_id '{instance_id}'。")
                    continue

                try:
                    file_list = ast.literal_eval(files_str)
                except (ValueError, SyntaxError):
                    print(f"错误：无法解析文件列表字符串: {files_str}")
                    continue

                # --- 阶段一：事前预防，构建初始内容列表 ---
                print("\n🔄 [阶段1] 正在预先读取并筛选文件，避免明显超长...")
                content_blocks = []
                files_for_prompt = []
                for rel_path in file_list:
                    full_path = os.path.join(LINUX_REPO_PATH, rel_path)
                    file_content = read_file_content(full_path)

                    # 构造临时prompt进行预检查
                    temp_agg_content = "\n\n".join(content_blocks + [file_content])
                    temp_prompt = OBTAIN_RELEVANT_FUNCTIONS_PROMPT.format(problem_statement=problem_statement, file_contents=temp_agg_content)

                    if num_tokens_from_messages(temp_prompt, "deepseek-chat") >= MAX_CONTEXT_LENGTH:
                        print(f"⚠️  预检查发现添加文件 {rel_path} 后可能超长，停止添加。")
                        break

                    content_blocks.append(file_content)
                    files_for_prompt.append(rel_path)
                    print(f"   ✅ 预添加文件: {rel_path}")

                if not content_blocks:
                    print(f"❌ 即使是第一个文件也可能超长，或所有文件读取失败，跳过此实例。")
                    continue

                # --- 阶段二：事后重试，循环调用LLM ---
                print(f"\n🔄 [阶段2] 开始请求LLM，包含失败重试机制...")
                max_retries = len(content_blocks)
                llm_result = None

                for attempt in range(max_retries):
                    print(f"   尝试 #{attempt + 1}/{max_retries}，使用 {len(content_blocks)} 个文件...")

                    # 构建当前尝试的prompt
                    current_contents_agg = "\n\n".join(content_blocks)
                    final_prompt = OBTAIN_RELEVANT_FUNCTIONS_PROMPT.format(
                        problem_statement=problem_statement,
                        file_contents=current_contents_agg
                    )

                    # 调用LLM
                    llm_result = query_llm(final_prompt)

                    # 分析结果
                    if llm_result == "TOKEN_LIMIT_EXCEEDED":
                        if len(content_blocks) > 1:
                            removed_file = files_for_prompt.pop()
                            content_blocks.pop()
                            print(f"   - API返回Token超限，移除最后一个文件 ({os.path.basename(removed_file)}) 后重试...")
                            time.sleep(1) # 短暂等待
                        else:
                            print("   - 即使只用一个文件也超限，无法再缩减。放弃。")
                            llm_result = None # 标记为最终失败
                            break
                    else:
                        # 成功或遇到其他不可重试的错误
                        break

                # --- 阶段三：处理并保存结果 ---
                print("\n🔄 [阶段3] 处理最终结果...")
                if llm_result:
                    model_found_locs = extract_code_blocks(llm_result)
                    model_found_locs_separated = extract_locs_for_files(model_found_locs, files_for_prompt)

                    # 保存原始结果
                    raw_output_csv = os.path.join(OUTPUT_DIR, "raw_outputs.csv")
                    file_exists = os.path.exists(raw_output_csv)
                    with open(raw_output_csv, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['instance_id', 'commit_hash', 'problem_statement', 'raw_llm_output', 'extracted_code_blocks'])
                        writer.writerow([instance_id, commit_hash, problem_statement, llm_result, model_found_locs])
                    print(f"   - 原始输出已保存到: {raw_output_csv}")

                    # 保存处理后的位置信息
                    locations_csv = os.path.join(OUTPUT_DIR, "locations.csv")
                    file_exists = os.path.exists(locations_csv)
                    with open(locations_csv, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['instance_id', 'file_path', 'location_type', 'location_name'])
                        for file_path, locs in model_found_locs_separated.items():
                            for loc in locs:
                                if ':' in loc:
                                    loc_type, loc_name = loc.split(':', 1)
                                    writer.writerow([instance_id, file_path, loc_type.strip(), loc_name.strip()])
                                else:
                                    writer.writerow([instance_id, file_path, 'unknown', loc.strip()])
                    print(f"   - 位置信息已保存到: {locations_csv}")

                else:
                    print(f"   - 未能从 LLM 获取 instance_id '{instance_id}' 的有效结果。")

                print("-" * 50)

            print("\n🎉 批处理完成!")

    except FileNotFoundError:
        print(f"错误：找不到 CSV 文件 {CSV_FILE_PATH}")
    except Exception as e:
        import traceback
        print(f"处理过程中发生未知错误: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
