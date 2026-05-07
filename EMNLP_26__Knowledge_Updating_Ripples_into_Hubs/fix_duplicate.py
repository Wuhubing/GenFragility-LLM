import re

file_path = "contents/method.tex"
with open(file_path, "r") as f:
    content = f.read()

# 找到 "For Llama-2, the probe uses cloze/completion prompts..." 
# 使用正则或 find 找到第二次出现的地方并删除
pattern = r"For Llama-2, the probe uses cloze/completion prompts rather than chat-style QA prompts\. This metric should therefore be interpreted as a mechanistic probe of last-layer, first-generation-step attention concentrated on the evaluated neighbor head-token span, not as a replacement for EPR\."

matches = list(re.finditer(pattern, content))
if len(matches) > 1:
    # 把第二次匹配及它后面的内容（如果在同一段落内）删掉
    start = matches[1].start()
    end = matches[1].end()
    # 删除该段落（假设它后面有 \n）
    content = content[:start] + content[end:]

with open(file_path, "w") as f:
    f.write(content)
