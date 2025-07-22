import argparse
import json
import os
import subprocess
import uuid
from tqdm import tqdm

# C语言解析需要
try:
    from clang.cindex import Index, CursorKind
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False
    print("警告：Clang库未找到，C语言解析功能将不可用。请运行 'pip install libclang'")


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        
        # 首先清理工作目录，强制重置到当前状态
        print("Cleaning up working directory...")
        subprocess.run(["git", "-C", repo_path, "clean", "-fd"], check=True)
        subprocess.run(["git", "-C", repo_path, "reset", "--hard", "HEAD"], check=True)
        
        # 然后执行checkout
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
        # 如果还是失败，尝试强制checkout
        try:
            print("Trying force checkout...")
            subprocess.run(["git", "-C", repo_path, "checkout", "-f", commit_id], check=True)
            print("Force checkout successful.")
        except subprocess.CalledProcessError as e2:
            print(f"Force checkout also failed: {e2}")
            raise e2
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e


def clone_repo(repo_name, local_path):
    """克隆一个指定的GitHub仓库到本地。"""
    print(f"Cloning repository from https://github.com/{repo_name}.git to {local_path}...")
    try:
        subprocess.run(["git", "clone", f"https://github.com/{repo_name}.git", local_path], check=True, capture_output=True)
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e.stderr.decode()}")
        exit(1)


def parse_c_file(file_path):
    """
    解析一个C文件，提取结构体、联合体、函数定义及其行号和文本。
    返回一个元组，格式与Python版本的 parse_python_file 保持一致。
    """
    struct_info = []
    function_info = []
    file_lines = []

    if not CLANG_AVAILABLE:
        return struct_info, function_info, file_lines

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
            file_lines = file_content.splitlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], [], []
        
    try:
        index = Index.create()
        tu = index.parse(file_path, args=['-I/usr/include', '-w'], unsaved_files=[(file_path, file_content)])
        
        for node in tu.cursor.walk_preorder():
            if node.location.file and node.location.file.name == file_path:
                start_line, end_line = node.extent.start.line, node.extent.end.line
                node_text = file_lines[start_line - 1 : end_line]
                
                if node.kind in [CursorKind.STRUCT_DECL, CursorKind.UNION_DECL] and node.is_definition():
                    struct_info.append({
                        "name": node.spelling or "anonymous",
                        "start_line": start_line,
                        "end_line": end_line,
                        "text": node_text
                    })
                elif node.kind == CursorKind.FUNCTION_DECL and node.is_definition():
                    function_info.append({
                        "name": node.spelling,
                        "start_line": start_line,
                        "end_line": end_line,
                        "text": node_text
                    })
    except Exception as e:
        print(f"Clang parsing error in file {file_path}: {e}")

    return struct_info, function_info, file_lines


def create_structure(directory_path):
    """
    通过解析C/H文件来创建仓库目录的结构。
    """
    print(f"开始分析目录结构: {directory_path}")
    structure = {}
    total_files = 0
    c_files = 0
    parsed_files = 0
    skipped_dirs = 0
    
    # 首先统计总文件数，用于进度显示
    print("正在统计文件总数...")
    for root, _, files in os.walk(directory_path):
        if '.git' in root:
            continue
        total_files += len(files)
    
    print(f"发现 {total_files} 个文件，开始解析...")
    
    processed_files = 0
    
    for root, dirs, files in os.walk(directory_path):
        if '.git' in root:
            skipped_dirs += 1
            continue
        
        # 显示当前处理的目录
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            print(f"\n📁 处理根目录")
        else:
            print(f"\n📁 处理目录: {relative_root}")
        
        # 为了与Python版本完全一致，我们在这里构建嵌套字典
        curr_struct = structure
        if relative_root != ".":
            for part in relative_root.split(os.sep):
                curr_struct = curr_struct.setdefault(part, {})
        
        print(f"   发现 {len(files)} 个文件")
        
        for file_name in files:
            processed_files += 1
            if file_name.endswith((".c", ".h")):
                c_files += 1
                file_path = os.path.join(root, file_name)
                print(f"   🔍 解析 C/H 文件: {file_name}", end="")
                
                try:
                    structs, functions, file_lines = parse_c_file(file_path)
                    # 这里的字典键名与Python版本保持一致，用 'classes' 对应 'structs'
                    curr_struct[file_name] = {
                        "classes": structs, # 用 'classes' 键来存储C的结构体信息
                        "functions": functions,
                        "text": file_lines,
                    }
                    
                    # 显示解析结果
                    print(f" ✅ (结构体: {len(structs)}, 函数: {len(functions)}, 行数: {len(file_lines)})")
                    parsed_files += 1
                    
                except Exception as e:
                    print(f" ❌ 解析失败: {e}")
                    curr_struct[file_name] = {
                        "classes": [],
                        "functions": [],
                        "text": [],
                    }
            else:
                curr_struct[file_name] = {}
                print(f"   📄 记录其他文件: {file_name}")
            
            # 显示进度
            if processed_files % 100 == 0 or processed_files == total_files:
                progress = (processed_files / total_files) * 100
                print(f"\n   📊 进度: {processed_files}/{total_files} ({progress:.1f}%)")
    
    # 最终统计
    print(f"\n" + "="*60)
    print(f"📈 解析完成!")
    print(f"   总文件数: {total_files}")
    print(f"   C/H文件数: {c_files}")
    print(f"   成功解析: {parsed_files}")
    print(f"   跳过的Git目录: {skipped_dirs}")
    print(f"   解析成功率: {(parsed_files/max(c_files,1)*100):.1f}%")
    print("="*60)
                
    return structure
def create_structure_for_subdirs(directory_path, target_subdirs=None):
    """
    只解析指定子目录的结构，提高效率
    """
    print(f"开始分析目录结构: {directory_path}")
    if target_subdirs:
        print(f"只解析子目录: {target_subdirs}")
    
    structure = {}
    total_files = 0
    c_files = 0
    parsed_files = 0
    skipped_dirs = 0
    
    for root, dirs, files in os.walk(directory_path):
        if '.git' in root:
            skipped_dirs += 1
            continue
        
        # 获取相对路径
        relative_root = os.path.relpath(root, directory_path)
        
        # 如果指定了目标子目录，检查当前路径是否在目标范围内
        if target_subdirs and relative_root != ".":
            # 检查是否是目标子目录或其子目录
            is_target = False
            for subdir in target_subdirs:
                if relative_root == subdir or relative_root.startswith(subdir + os.sep):
                    is_target = True
                    break
            
            if not is_target:
                continue  # 跳过非目标子目录
        
        # 显示当前处理的目录
        if relative_root == ".":
            print(f"\n📁 处理根目录")
        else:
            print(f"\n📁 处理目录: {relative_root}")
        
        # 构建嵌套字典
        curr_struct = structure
        if relative_root != ".":
            for part in relative_root.split(os.sep):
                curr_struct = curr_struct.setdefault(part, {})
        
        print(f"   发现 {len(files)} 个文件")
        
        for file_name in files:
            total_files += 1
            if file_name.endswith((".c", ".h")):
                c_files += 1
                file_path = os.path.join(root, file_name)
                print(f"   🔍 解析 C/H 文件: {file_name}", end="")
                
                try:
                    structs, functions, file_lines = parse_c_file(file_path)
                    curr_struct[file_name] = {
                        "classes": structs,
                        "functions": functions,
                        "text": file_lines,
                    }
                    
                    print(f" ✅ (结构体: {len(structs)}, 函数: {len(functions)}, 行数: {len(file_lines)})")
                    parsed_files += 1
                    
                except Exception as e:
                    print(f" ❌ 解析失败: {e}")
                    curr_struct[file_name] = {
                        "classes": [],
                        "functions": [],
                        "text": [],
                    }
            else:
                curr_struct[file_name] = {}
                print(f"   📄 记录其他文件: {file_name}")
            
            # 显示进度
            if total_files % 100 == 0:
                print(f"\n   📊 已处理: {total_files} 个文件")
    
    # 最终统计
    print(f"\n" + "="*60)
    print(f"📈 解析完成!")
    print(f"   总文件数: {total_files}")
    print(f"   C/H文件数: {c_files}")
    print(f"   成功解析: {parsed_files}")
    print(f"   跳过的Git目录: {skipped_dirs}")
    if target_subdirs:
        print(f"   目标子目录: {target_subdirs}")
    print("="*60)
                
    return structure


def get_project_structure_from_scratch(repo_name, commit_id, instance_id, repo_playground,subdirs=None):
    """
    【C语言版】
    与Python版本逻辑完全相同的顶层接口。
    """
    # 1. 创建临时的、带UUID的沙盒文件夹
    temp_playground = os.path.join(repo_playground, str(uuid.uuid4()))
    assert not os.path.exists(temp_playground), f"{temp_playground} already exists"
    os.makedirs(temp_playground)
    
    # 2. 在沙盒中克隆仓库
    # local_folder_name = repo_name.split('/')[-1]
    # local_repo_path = os.path.join(temp_playground, local_folder_name)
    # clone_repo(repo_name, local_repo_path)

    local_repo_path = "/root/Agentless/linux"  # 假设你的Linux仓库在这里
    checkout_commit(local_repo_path, commit_id)
    # 3. 检出commit
    # if not checkout_commit(local_repo_path, commit_id):
    #     print(f"检出失败，清理并退出对 {instance_id} 的分析。")
    # subprocess.run(["rm", "-rf", temp_playground], check=True)
        # return None

    # 4. 分析结构
    print("开始分析代码结构...")
    # structure = create_structure(local_repo_path)
    structure = create_structure_for_subdirs(local_repo_path, subdirs)
    # 5. 清理沙盒
    print(f"清理临时文件夹: {temp_playground}")
    subprocess.run(["rm", "-rf", temp_playground], check=True)
    
    # 6. 返回与Python版本完全一致的结果
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d

