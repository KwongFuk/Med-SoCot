import json
import argparse

def remove_entries_with_low_comprehensive(file_path):
    updated_data = []
    deleted_indices = []

    # 逐行读取 .jsonl 文件
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            entry = json.loads(line.strip())
            if 'prediction_scores' in entry:
                # 检查 comprehensive 是否小于 50
                if entry['prediction_scores'][0].get('comprehensive', 0) < 70:
                    # 删除指定的键
                    keys_to_remove = ["sample_predictions", "prediction_scores", "inference_time", "all_output"]
                    for key in keys_to_remove:
                        entry.pop(key, None)
                    deleted_indices.append(index)  # 记录被删除的序号
            updated_data.append(entry)  # 将更新后的数据添加到列表

    # 将更新后的数据写回到原文件
    with open(file_path, 'w') as file:
        for entry in updated_data:
            file.write(json.dumps(entry) + '\n')

    return deleted_indices

# 配置命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove entries with low comprehensive scores from a JSONL file.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL file")

    args = parser.parse_args()
    deleted_indices = remove_entries_with_low_comprehensive(args.file_path)
    print("Deleted indices:", deleted_indices)
