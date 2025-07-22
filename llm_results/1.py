#!/usr/bin/env python3
"""
合并locations.csv文件的脚本
- 将相同instance_id的记录合并
- 将相同文件路径的记录合并
- 生成更清晰的输出文件
"""

import csv
import os
from collections import defaultdict
import json

def load_locations_csv(file_path):
    """加载locations.csv文件"""
    print(f"📖 正在读取文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    print(f"✅ 成功读取 {len(data)} 条记录")
    return data

def merge_by_instance_and_file(data):
    """按instance_id和file_path分组合并"""
    print(f"🔄 开始按instance_id和file_path分组...")
    
    # 使用嵌套字典: instance_id -> file_path -> locations
    grouped = defaultdict(lambda: defaultdict(list))
    
    for row in data:
        instance_id = row['instance_id']
        file_path = row['file_path']
        location_info = {
            'type': row['location_type'],
            'name': row['location_name']
        }
        grouped[instance_id][file_path].append(location_info)
    
    # 统计信息
    total_instances = len(grouped)
    total_files = sum(len(files) for files in grouped.values())
    total_locations = sum(
        sum(len(locations) for locations in files.values()) 
        for files in grouped.values()
    )
    
    print(f"📊 分组统计:")
    print(f"   - 唯一instance_id: {total_instances} 个")
    print(f"   - 唯一文件: {total_files} 个")
    print(f"   - 总位置数: {total_locations} 个")
    
    return grouped

def save_merged_csv(grouped_data, output_file):
    """保存合并后的CSV文件"""
    print(f"💾 正在保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['instance_id', 'file_path', 'location_count', 'functions', 'structs', 'variables', 'macros', 'others'])
        
        for instance_id, files in grouped_data.items():
            for file_path, locations in files.items():
                # 按类型分组位置
                functions = []
                structs = []
                variables = []
                macros = []
                others = []
                
                for loc in locations:
                    loc_type = loc['type'].lower()
                    loc_name = loc['name']
                    
                    if 'function' in loc_type:
                        functions.append(loc_name)
                    elif 'struct' in loc_type:
                        structs.append(loc_name)
                    elif 'variable' in loc_type or 'var' in loc_type:
                        variables.append(loc_name)
                    elif 'macro' in loc_type:
                        macros.append(loc_name)
                    else:
                        others.append(f"{loc_type}:{loc_name}")
                
                # 去重并排序
                functions = sorted(list(set(functions)))
                structs = sorted(list(set(structs)))
                variables = sorted(list(set(variables)))
                macros = sorted(list(set(macros)))
                others = sorted(list(set(others)))
                
                total_count = len(functions) + len(structs) + len(variables) + len(macros) + len(others)
                
                # 写入行数据
                writer.writerow([
                    instance_id,
                    file_path,
                    total_count,
                    '; '.join(functions) if functions else '',
                    '; '.join(structs) if structs else '',
                    '; '.join(variables) if variables else '',
                    '; '.join(macros) if macros else '',
                    '; '.join(others) if others else ''
                ])
    
    print(f"✅ 合并后的CSV文件已保存")

def save_summary_json(grouped_data, output_file):
    """保存详细的JSON格式摘要"""
    print(f"💾 正在保存JSON摘要到: {output_file}")
    
    summary = {}
    
    for instance_id, files in grouped_data.items():
        instance_summary = {
            'total_files': len(files),
            'files': {}
        }
        
        for file_path, locations in files.items():
            # 按类型统计
            type_counts = defaultdict(int)
            type_details = defaultdict(list)
            
            for loc in locations:
                loc_type = loc['type']
                loc_name = loc['name']
                type_counts[loc_type] += 1
                type_details[loc_type].append(loc_name)
            
            # 去重
            for loc_type in type_details:
                type_details[loc_type] = sorted(list(set(type_details[loc_type])))
            
            instance_summary['files'][file_path] = {
                'total_locations': len(locations),
                'type_counts': dict(type_counts),
                'locations': dict(type_details)
            }
        
        summary[instance_id] = instance_summary
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON摘要文件已保存")

def generate_statistics_report(grouped_data, output_file):
    """生成统计报告"""
    print(f"📊 正在生成统计报告: {output_file}")
    
    # 统计信息
    total_instances = len(grouped_data)
    all_files = set()
    all_location_types = defaultdict(int)
    instances_by_file_count = defaultdict(int)
    files_by_location_count = defaultdict(int)
    
    for instance_id, files in grouped_data.items():
        instances_by_file_count[len(files)] += 1
        
        for file_path, locations in files.items():
            all_files.add(file_path)
            files_by_location_count[len(locations)] += 1
            
            for loc in locations:
                all_location_types[loc['type']] += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 位置数据统计报告\n\n")
        
        f.write(f"## 总体统计\n")
        f.write(f"- 总instance数量: {total_instances}\n")
        f.write(f"- 唯一文件数量: {len(all_files)}\n")
        f.write(f"- 总位置数量: {sum(all_location_types.values())}\n\n")
        
        f.write(f"## 按位置类型统计\n")
        for loc_type, count in sorted(all_location_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {loc_type}: {count} 个\n")
        f.write("\n")
        
        f.write(f"## 按文件数量分布\n")
        for file_count, instance_count in sorted(instances_by_file_count.items()):
            f.write(f"- {file_count} 个文件的instance: {instance_count} 个\n")
        f.write("\n")
        
        f.write(f"## 文件中位置数量分布\n")
        for loc_count, file_count in sorted(files_by_location_count.items()):
            f.write(f"- {loc_count} 个位置的文件: {file_count} 个\n")
        f.write("\n")
        
        f.write(f"## 最常出现的文件 (Top 10)\n")
        file_frequency = defaultdict(int)
        for files in grouped_data.values():
            for file_path in files.keys():
                file_frequency[file_path] += 1
        
        top_files = sorted(file_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        for file_path, frequency in top_files:
            f.write(f"- {file_path}: 出现在 {frequency} 个instance中\n")
    
    print(f"✅ 统计报告已保存")

def main():
    """主函数"""
    print("🚀 开始合并locations.csv文件")
    print("=" * 50)
    
    # 输入和输出文件路径
    input_file = "locations.csv"
    output_csv = "final.csv"
    output_json = "locations_summary.json"
    output_report = "locations_statistics.md"
    
    # 1. 加载数据
    data = load_locations_csv(input_file)
    if data is None:
        return
    
    # 2. 分组合并
    grouped_data = merge_by_instance_and_file(data)
    
    # 3. 保存合并后的CSV
    save_merged_csv(grouped_data, output_csv)
    
    # 4. 保存JSON摘要
    save_summary_json(grouped_data, output_json)
    
    # 5. 生成统计报告
    generate_statistics_report(grouped_data, output_report)
    
    print("\n🎉 处理完成!")
    print("📁 输出文件:")
    print(f"   📊 合并后的CSV: {output_csv}")
    print(f"   📋 详细JSON摘要: {output_json}")
    print(f"   📈 统计报告: {output_report}")

if __name__ == "__main__":
    main()
