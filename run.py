# 1. 定义文件路径（修改为你的txt路径）
input_path = "sensitivity_log2.txt"
output_path = "sensitivity_saved.txt"  # 输出新文件，如需覆盖原文件可设为同路径
target_prefix = "🔍评估"

# 2. 逐行处理：过滤目标前缀行 + 过滤空行
with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 先去除首尾空白字符，用于双重判断
        stripped_line = line.strip()
        # 条件：1. 非空行（stripped_line不为空）  2. 不以目标前缀开头
        if stripped_line and not stripped_line.startswith(target_prefix):
            outfile.write(line)  # 写入原始行，保留有效行的原始格式

print("处理完成，结果已保存到", output_path)