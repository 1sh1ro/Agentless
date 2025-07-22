import pandas as pd
import json

def extract_data_from_jsonl(file_path):
    """
    从一个 jsonl 文件中逐行读取，并提取 'instance_id' 和 'final_ranked_files'。
    """
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 跳过空行
                if not line.strip():
                    continue
                try:
                    # 解析当前行
                    json_obj = json.loads(line)
                    
                    # 使用 .get() 方法安全地提取数据，如果键不存在则返回 None
                    instance_id = json_obj.get('instance_id')
                    final_ranked_files = json_obj.get('final_ranked_files')
                    
                    data_list.append({
                        'instance_id': instance_id,
                        'final_ranked_files': final_ranked_files
                    })
                except json.JSONDecodeError:
                    print(f"警告：文件 '{file_path}' 中有一行不是有效的 JSON，已跳过。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。")
        return None
        
    return data_list

# --- 在这里修改你的文件名 ---
file1_path = '/root/Agentless/agentless/results/linux_final2/loc_outputs.jsonl'
file2_path = '/root/Agentless/root/Agentless/agentless/results/linux_final3/loc_outputs.jsonl'
output_path = 'merged_output.csv'
# -----------------------------

# 分别从两个文件中提取数据
data1 = extract_data_from_jsonl(file1_path)
data2 = extract_data_from_jsonl(file2_path)

if data1 is not None and data2 is not None:
    # 合并两个文件的数据列表
    all_data = data1 + data2

    # 将数据列表转换为 pandas DataFrame
    df = pd.DataFrame(all_data)

    # 将 DataFrame 保存为 CSV 文件
    # index=False 表示不将 DataFrame 的索引写入文件
    # encoding='utf-8-sig' 确保 Excel 等软件能正确识别 UTF-8 编码
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"处理完成！数据已合并并保存到 '{output_path}'。")
