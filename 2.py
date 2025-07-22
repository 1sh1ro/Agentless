import csv
import json
import os
import subprocess
import requests
import ast
from clang.cindex import Index, CursorKind

# --- 1. 配置区域 ---

# 请将此路径修改为您本地的 Linux 内核代码仓库路径
LINUX_REPO_PATH = "/root/Agentless/linux/" 

# 输入文件
CSV_FILE_PATH = 'merged_output.csv'  # 您的 CSV 文件名
JSONL_FILE_PATH = '2.jsonl'

# 大模型 API 配置 (请填入您的真实信息)
API_URL = "https://api.deepseek.com/chat/completions"
API_KEY = ""# use yooooooooooooooooooooooooooooour key

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
```
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
```

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

def checkout_commit(repo_path, commit_hash):
    """在指定的仓库路径中切换到特定的 commit。"""
    if not os.path.isdir(repo_path):
        print(f"错误：指定的 Linux 仓库路径不存在: {repo_path}")
        return False

    print(f"正在切换到 commit: {commit_hash} ...")
    
    # 检查并清理Git锁文件
    lock_file = os.path.join(repo_path, '.git', 'index.lock')
    if os.path.exists(lock_file):
        print(f"⚠️  发现Git锁文件，正在清理: {lock_file}")
        try:
            os.remove(lock_file)
            print(f"✅ 成功删除Git锁文件")
        except Exception as e:
            print(f"❌ 删除锁文件失败: {e}")
            return False
    
    try:
        # 先清理工作区，避免 checkout 失败
        print(f"🔄 重置工作区到HEAD...")
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, check=True, capture_output=True, text=True)
        
        # 清理所有未被追踪的文件和目录
        print(f"🧹 清理未追踪的文件...")
        subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, check=True, capture_output=True, text=True)
        
        # 确保没有正在进行的合并或变基操作
        merge_head = os.path.join(repo_path, '.git', 'MERGE_HEAD')
        rebase_apply = os.path.join(repo_path, '.git', 'rebase-apply')
        rebase_merge = os.path.join(repo_path, '.git', 'rebase-merge')
        
        if os.path.exists(merge_head):
            print(f"🔄 检测到未完成的合并，正在中止...")
            subprocess.run(['git', 'merge', '--abort'], cwd=repo_path, check=False, capture_output=True, text=True)
            
        if os.path.exists(rebase_apply) or os.path.exists(rebase_merge):
            print(f"🔄 检测到未完成的变基，正在中止...")
            subprocess.run(['git', 'rebase', '--abort'], cwd=repo_path, check=False, capture_output=True, text=True)
        
        # 现在可以安全地切换到目标 commit
        print(f"📦 切换到目标commit...")
        result = subprocess.run(['git', 'checkout', commit_hash], cwd=repo_path, check=True, capture_output=True, text=True)
        print("✅ 切换 commit 成功。")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 切换到 commit {commit_hash} 失败。")
        print(f"Git 命令输出:\n{e.stderr}")
        
        # 如果还是失败，尝试更激进的清理
        print(f"🔄 尝试更激进的清理方法...")
        try:
            # 删除可能存在的其他锁文件
            lock_files = [
                os.path.join(repo_path, '.git', 'index.lock'),
                os.path.join(repo_path, '.git', 'HEAD.lock'),
                os.path.join(repo_path, '.git', 'config.lock'),
                os.path.join(repo_path, '.git', 'refs', 'heads', 'master.lock'),
                os.path.join(repo_path, '.git', 'refs', 'heads', 'main.lock'),
            ]
            
            for lock_file in lock_files:
                if os.path.exists(lock_file):
                    print(f"🗑️  删除锁文件: {lock_file}")
                    os.remove(lock_file)
            
            # 再次尝试切换
            subprocess.run(['git', 'checkout', commit_hash], cwd=repo_path, check=True, capture_output=True, text=True)
            print("✅ 激进清理后切换成功。")
            return True
            
        except Exception as cleanup_error:
            print(f"❌ 激进清理也失败: {cleanup_error}")
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
        # 对于C/C++文件，使用libclang解析
        if file_path.endswith(('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp')):
            print(f"🔧 识别为C/C++文件，使用libclang解析")
            return extract_c_structures(file_path)
        else:
            print(f"📄 识别为其他类型文件，使用文本模式")
            # 对于其他文件，读取前1000行避免超限
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if len(lines) > 1000:
                    content = ''.join(lines[:1000]) + "\n... (文件被截断，仅显示前1000行) ..."
                    print(f"📏 文件过长({len(lines)}行)，截断到1000行")
                else:
                    content = ''.join(lines)
                    print(f"📄 读取完整文件({len(lines)}行)")
            
            return FILE_CONTENT_TEMPLATE.format(
                file_name=file_path,
                file_content=content
            )
    except Exception as e:
        print(f"❌ 文件处理失败: {e}")
        return f"# 读取文件失败: {file_path}, 错误: {e}\n"

def extract_c_structures(file_path):
    """使用libclang提取C/C++文件的结构化信息"""
    print(f"📁 开始使用libclang解析文件: {file_path}")
    try:
        index = Index.create()
        print(f"🔧 创建libclang索引成功")
        
        # 尝试解析文件
        translation_unit = index.parse(file_path)
        print(f"📖 解析翻译单元成功")
        
        # 检查是否有诊断信息（错误/警告）
        diagnostics = list(translation_unit.diagnostics)
        if diagnostics:
            print(f"⚠️  发现 {len(diagnostics)} 个诊断信息:")
            for diag in diagnostics[:5]:  # 只显示前5个
                print(f"   - {diag.severity}: {diag.spelling}")
        
        structures = []
        structures.append(f"### File: {file_path} ###")
        
        function_count = 0
        struct_count = 0
        var_count = 0
        macro_count = 0
        
        def visit_node(node, level=0):
            nonlocal function_count, struct_count, var_count, macro_count
            indent = "  " * level
            
            # 提取函数定义
            if node.kind == CursorKind.FUNCTION_DECL:
                func_name = node.spelling
                if func_name and node.is_definition():
                    function_count += 1
                    print(f"🔍 发现函数: {func_name}")
                    # 获取函数签名
                    try:
                        start = node.extent.start
                        end = node.extent.end
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            if start.line <= len(lines) and end.line <= len(lines):
                                func_lines = lines[start.line-1:end.line]
                                # 只取函数声明部分（前几行）
                                func_decl = ''.join(func_lines[:min(3, len(func_lines))])
                                structures.append(f"{indent}FUNCTION: {func_name}")
                                structures.append(f"{indent}  {func_decl.strip()}")
                    except Exception as e:
                        print(f"   ⚠️ 获取函数 {func_name} 的源代码失败: {e}")
                        structures.append(f"{indent}FUNCTION: {func_name}")
            
            # 提取结构体/联合体定义
         #   elif node.kind in [CursorKind.STRUCT_DECL, CursorKind.UNION_DECL]:
          #      struct_name = node.spelling
           #     if struct_name:
            #        struct_count += 1
             #       print(f"🏗️  发现结构体: {struct_name}")
              #      structures.append(f"{indent}STRUCT: {struct_name}")
                    
            # 提取全局变量
            #elif node.kind == CursorKind.VAR_DECL and level == 0:
             #   var_name = node.spelling
              #  if var_name:
               #     var_count += 1
                #    print(f"🌐 发现全局变量: {var_name}")
                 #   structures.append(f"{indent}GLOBAL_VAR: {var_name}")
            
            # 提取宏定义
            #elif node.kind == CursorKind.MACRO_DEFINITION:
             #   macro_name = node.spelling
              #  if macro_name:
               #     macro_count += 1
                #    print(f"🔧 发现宏: {macro_name}")
                 #   structures.append(f"{indent}MACRO: {macro_name}")
            
            # 递归遍历子节点（但限制深度）
            if level < 3:
                for child in node.get_children():
                    visit_node(child, level + 1)
        
        # 开始遍历
        print(f"🔄 开始遍历AST节点...")
        visit_node(translation_unit.cursor)
        
        # 输出统计信息
        print(f"📊 解析完成! 统计结果:")
        print(f"   - 函数: {function_count} 个")
        print(f"   - 结构体: {struct_count} 个") 
        print(f"   - 全局变量: {var_count} 个")
        print(f"   - 宏定义: {macro_count} 个")
        
        # 如果没有提取到结构信息，回退到简单的文本方式
        # if len(structures) <= 1:
        #     print(f"⚠️  没有提取到任何结构化信息，回退到文本模式")
        #     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        #         lines = f.readlines()
        #         content = ''.join(lines[:500])  # 只读取前500行
        #         return FILE_CONTENT_TEMPLATE.format(
        #             file_name=file_path,
        #             file_content=content + "\n... (使用libclang解析失败，显示前500行)"
        #         )
        
        # print(f"✅ libclang解析成功，提取到 {len(structures)-1} 项结构化信息")
        return '\n'.join(structures) + '\n'
        
    except Exception as e:
        print(f"❌ libclang解析失败: {e}")
        # libclang解析失败时的回退方案
        try:
            print(f"🔄 回退到简单文本读取模式...")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                content = ''.join(lines[:500])  # 只读取前500行
                print(f"📄 成功读取文件前500行，共 {len(lines)} 行")
                return FILE_CONTENT_TEMPLATE.format(
                    file_name=file_path,
                    file_content=content + f"\n... (libclang解析失败: {e}，显示前500行)"
                )
        except Exception as read_error:
            print(f"❌ 文件读取也失败: {read_error}")
            return f"# 读取文件失败: {file_path}, 错误: {e}\n"

def query_llm(prompt):
    """向大模型发送请求并返回结果。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 使用正确的DeepSeek API格式
    payload = {
        "model": "deepseek-chat",  # 修正：使用正确的模型名称
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in code analysis and debugging."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": False  # 明确指定不使用流式输出
    }

    print("正在向大模型发送请求...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        # 更详细的错误处理
        if response.status_code != 200:
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return None
            
        response.raise_for_status()

        response_data = response.json()
        # 适配不同的API返回格式
        if "choices" in response_data and response_data["choices"]:
            # 支持messages格式的API
            if "message" in response_data["choices"][0]:
                result_text = response_data["choices"][0]["message"].get("content", "")
            # 支持text格式的API
            else:
                result_text = response_data["choices"][0].get("text", "")
            return result_text.strip()
        else:
            print(f"API响应格式异常: {response_data}")
            return f"未能从 API 响应中提取有效内容: {response.text}"

    except requests.exceptions.RequestException as e:
        print(f"错误：API 请求失败: {e}")
        return None


def num_tokens_from_messages(message, model_name):
    """估算token数量的简单实现（原始版本使用更复杂的计算）"""
    # 简单估算：一般来说，1个token约等于4个字符
    return len(message) // 4

def extract_code_blocks(text):
    """从LLM输出中提取代码块（与原始版本保持一致）"""
    if '```' in text:
        # 查找第一个和最后一个```之间的内容
        parts = text.split('```')
        if len(parts) >= 3:
            return parts[1].strip()
    return text.strip()

def extract_locs_for_files(model_found_locs, file_names, keep_old_order=False):
    """解析定位结果并按文件分组（适配libclang结构化输出）"""
    result = {}
    lines = model_found_locs.split('\n')
    current_file = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检查是否是文件路径
        if any(line.endswith(ext) for ext in ['.c', '.h', '.py', '.cpp', '.cc', '.cxx', '.hpp']) or '/' in line:
            current_file = line
            if current_file not in result:
                result[current_file] = []
        elif current_file and line.startswith(('function:', 'struct:', 'variable:', 'macro:', 'class:')):
            result[current_file].append(line)
    
    return result

def cleanup_git_locks(repo_path):
    """清理Git仓库中的所有锁文件"""
    print(f"🧹 开始清理Git锁文件...")
    
    lock_patterns = [
        '.git/index.lock',
        '.git/HEAD.lock',
        '.git/config.lock',
        '.git/refs/heads/*.lock',
        '.git/refs/remotes/*/*.lock',
        '.git/objects/pack/*.lock',
    ]
    
    import glob
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

# --- 4. 主执行逻辑 ---

def main():
    """主函数，协调整个流程。"""
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    # 预先清理Git锁文件，避免后续问题
    print(f"🔧 预检查Git仓库状态...")
    if os.path.exists(LINUX_REPO_PATH):
        cleanup_git_locks(LINUX_REPO_PATH)
    else:
        print(f"⚠️  Linux仓库路径不存在: {LINUX_REPO_PATH}")

    # 检查现有CSV文件状态
    raw_output_csv = os.path.join(OUTPUT_DIR, "raw_outputs.csv")
    locations_csv = os.path.join(OUTPUT_DIR, "locations.csv")
    
    if os.path.exists(raw_output_csv):
        with open(raw_output_csv, 'r', encoding='utf-8') as f:
            existing_raw_count = sum(1 for line in f) - 1  # 减去表头
        print(f"📄 发现现有原始输出文件，已有 {existing_raw_count} 条记录")
    else:
        print(f"📄 原始输出文件不存在，将创建新文件")
        
    if os.path.exists(locations_csv):
        with open(locations_csv, 'r', encoding='utf-8') as f:
            existing_loc_count = sum(1 for line in f) - 1  # 减去表头
        print(f"📄 发现现有位置信息文件，已有 {existing_loc_count} 条记录")
    else:
        print(f"📄 位置信息文件不存在，将创建新文件")

    # 加载数据集
    dataset = load_dataset_to_dict(JSONL_FILE_PATH)
    if not dataset:
        return

    # 读取并处理 CSV 文件
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # 跳过表头
            
            # 将所有行读入列表
            all_rows = list(reader)
            total_rows = len(all_rows)
            
            # 从第141行开始处理（索引从0开始，所以是140）
            start_index = 128  # 第141行对应索引140
            
            print(f"开始处理 CSV 文件: {CSV_FILE_PATH}")
            print(f"总共 {total_rows} 行数据，从第 {start_index + 1} 行开始处理")
            print("-" * 50)

            for i in range(start_index, total_rows):
                row = all_rows[i]
                instance_id, files_str = row
                print(f"\n处理第 {i+1} 行: instance_id = {instance_id} (进度: {i+1-start_index}/{total_rows-start_index})")

                # 从数据集中查找对应信息
                if instance_id not in dataset:
                    print(f"警告：在 {JSONL_FILE_PATH} 中找不到 instance_id '{instance_id}'，跳过此行。")
                    continue

                bug_data = dataset[instance_id]
                commit_hash = bug_data.get('commit')
                problem_statement = bug_data.get('report')

                if not commit_hash or not problem_statement:
                    print(f"警告：instance_id '{instance_id}' 的数据不完整，跳过。")
                    continue

                # 1. 切换 Git Commit（带重试机制）
                max_retries = 3
                retry_count = 0
                checkout_success = False
                
                while retry_count < max_retries and not checkout_success:
                    if retry_count > 0:
                        print(f"🔄 第 {retry_count + 1} 次尝试切换commit...")
                        # 等待一下再重试
                        import time
                        time.sleep(2)
                        # 重新清理锁文件
                        cleanup_git_locks(LINUX_REPO_PATH)
                    
                    checkout_success = checkout_commit(LINUX_REPO_PATH, commit_hash)
                    retry_count += 1
                
                if not checkout_success:
                    print(f"❌ 经过 {max_retries} 次尝试仍无法切换到 commit，跳过 instance_id '{instance_id}'。")
                    continue

                # 2. 准备文件内容
                try:
                    # 使用 ast.literal_eval 安全地解析字符串列表
                    file_list = ast.literal_eval(files_str)
                    print(f"📋 需要分析 {len(file_list)} 个文件:")
                    for idx, f in enumerate(file_list, 1):
                        print(f"   {idx}. {f}")
                except (ValueError, SyntaxError):
                    print(f"错误：无法解析文件列表字符串: {files_str}")
                    continue

                all_file_contents = []
                print("\n🔄 正在读取相关文件...")
                for rel_path in file_list:
                    full_path = os.path.join(LINUX_REPO_PATH, rel_path)
                    file_content = read_file_content(full_path)
                    all_file_contents.append(file_content)
                    print()  # 空行分隔每个文件的输出

                print(f"📊 文件读取完成，共处理 {len(all_file_contents)} 个文件")

                # 3. 构建初始Prompt（使用换行分隔以便更好地控制长度）
                file_contents_agg = "\n\n".join(all_file_contents)
                final_prompt = OBTAIN_RELEVANT_FUNCTIONS_PROMPT.format(
                    problem_statement=problem_statement,
                    file_contents=file_contents_agg
                )
                
                initial_token_count = num_tokens_from_messages(final_prompt, "deepseek-chat")
                print(f"📏 初始Prompt长度: {initial_token_count} tokens (限制: {MAX_CONTEXT_LENGTH})")

                # 4. Token长度控制（使用正确的模型名称）
                def message_too_long(message):
                    return num_tokens_from_messages(message, "deepseek-chat") >= MAX_CONTEXT_LENGTH

                reduction_count = 0
                while message_too_long(final_prompt) and len(all_file_contents) > 1:
                    reduction_count += 1
                    print(f"⚠️  消息过长，第{reduction_count}次减少文件数量: {len(all_file_contents)} -> {len(all_file_contents)-1}")
                    all_file_contents = all_file_contents[:-1]
                    file_contents_agg = "\n\n".join(all_file_contents)
                    final_prompt = OBTAIN_RELEVANT_FUNCTIONS_PROMPT.format(
                        problem_statement=problem_statement,
                        file_contents=file_contents_agg
                    )
                    current_tokens = num_tokens_from_messages(final_prompt, "deepseek-chat")
                    print(f"   📏 当前长度: {current_tokens} tokens")

                if message_too_long(final_prompt):
                    final_tokens = num_tokens_from_messages(final_prompt, "deepseek-chat")
                    print(f"❌ 即使减少文件数量，消息仍然太长 ({final_tokens} tokens)，跳过 instance_id '{instance_id}'")
                    continue
                
                if reduction_count > 0:
                    final_tokens = num_tokens_from_messages(final_prompt, "deepseek-chat")
                    print(f"✅ Token长度控制完成，最终使用 {len(all_file_contents)} 个文件，{final_tokens} tokens")
                else:
                    print(f"✅ Token长度符合要求，使用全部 {len(all_file_contents)} 个文件")

                # 5. 查询大模型
                llm_result = query_llm(final_prompt)

                # 6. 结果后处理（保存为CSV格式）
                if llm_result:
                    # 提取代码块
                    model_found_locs = extract_code_blocks(llm_result)
                    # 按文件分组
                    model_found_locs_separated = extract_locs_for_files(
                        model_found_locs, file_list, keep_old_order=False
                    )
                    
                    # 保存原始结果到CSV
                    raw_output_csv = os.path.join(OUTPUT_DIR, "raw_outputs.csv")
                    try:
                        # 检查文件是否存在，如果不存在则写入表头
                        file_exists = os.path.exists(raw_output_csv)
                        with open(raw_output_csv, 'a', encoding='utf-8', newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(['instance_id', 'commit_hash', 'problem_statement', 'raw_llm_output', 'extracted_code_blocks'])
                            writer.writerow([instance_id, commit_hash, problem_statement, llm_result, model_found_locs])
                        print(f"原始输出保存到CSV: {raw_output_csv}")
                    except IOError as e:
                        print(f"错误：无法写入原始输出CSV文件: {e}")
                    
                    # 保存处理后的结果到CSV
                    locations_csv = os.path.join(OUTPUT_DIR, "locations.csv")
                    try:
                        # 检查文件是否存在，如果不存在则写入表头
                        file_exists = os.path.exists(locations_csv)
                        with open(locations_csv, 'a', encoding='utf-8', newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(['instance_id', 'file_path', 'location_type', 'location_name'])
                            
                            # 为每个文件的每个位置写入一行
                            for file_path, locs in model_found_locs_separated.items():
                                for loc in locs:
                                    # 解析位置类型和名称
                                    if ':' in loc:
                                        loc_type, loc_name = loc.split(':', 1)
                                        loc_type = loc_type.strip()
                                        loc_name = loc_name.strip()
                                    else:
                                        loc_type = 'unknown'
                                        loc_name = loc.strip()
                                    writer.writerow([instance_id, file_path, loc_type, loc_name])
                        print(f"位置信息保存到CSV: {locations_csv}")
                    except IOError as e:
                        print(f"错误：无法写入位置信息CSV文件: {e}")
                    
                    # 保存进度信息
                    progress_file = os.path.join(OUTPUT_DIR, "progress.txt")
                    try:
                        with open(progress_file, 'w', encoding='utf-8') as f:
                            f.write(f"最后处理的行: {i+1}\n")
                            f.write(f"最后处理的instance_id: {instance_id}\n")
                            f.write(f"处理时间: {__import__('datetime').datetime.now()}\n")
                    except:
                        pass  # 进度文件写入失败不影响主流程
                        
                else:
                    print(f"未能从 LLM 获取 instance_id '{instance_id}' 的结果。")

                print("-" * 50)
            
            # 处理完成统计
            processed_count = total_rows - start_index
            print(f"\n🎉 批处理完成!")
            print(f"📊 处理统计:")
            print(f"   - 起始行: 第{start_index + 1}行")
            print(f"   - 结束行: 第{total_rows}行")
            print(f"   - 总处理: {processed_count} 条记录")
            print(f"   - 输出文件:")
            print(f"     * {raw_output_csv}")
            print(f"     * {locations_csv}")

    except FileNotFoundError:
        print(f"错误：找不到 CSV 文件 {CSV_FILE_PATH}")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")


if __name__ == '__main__':
    main()
