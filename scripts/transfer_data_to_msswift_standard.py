import json

def convert_orpo_format(input_file, output_file):
    """
    将自定义原子事实提取数据转换为标准 ORPO messages 格式
    
    注意：chosen 是字符串，rejected 是字典（dict）
    """
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行 JSON 解析失败: {e}")
                continue
            
            # 构建标准 messages 格式
            messages = list(data['messages'])  # 复制一份
            
            # 处理 chosen（它是 JSON 字符串）
            chosen_str = data.get('chosen', '{}')
            try:
                chosen_obj = json.loads(chosen_str) if isinstance(chosen_str, str) else chosen_str
                chosen_content = json.dumps(chosen_obj, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                chosen_content = str(chosen_str)
            
            # 添加 assistant 的正确回复
            messages.append({
                "role": "assistant",
                "content": chosen_content
            })
            
            # 处理 rejected（它是 dict，不是字符串）
            rejected_data = data.get('rejected', {})
            if isinstance(rejected_data, dict):
                # 提取 thinking 和实际输出
                # 如果 rejected 有 thinking 字段，说明是思考过程
                if 'thinking' in rejected_data:
                    # 尝试从 thinking 中提取最终的 JSON 输出
                    thinking_text = rejected_data.get('thinking', '')
                    # 寻找 thinking 末尾的 JSON 对象
                    import re
                    json_match = re.search(r'\n(\{.*\})\s*$', thinking_text, re.DOTALL)
                    if json_match:
                        rejected_content = json_match.group(1)
                    else:
                        # 如果没有找到，就用整个 thinking 作为 rejected
                        rejected_content = thinking_text
                else:
                    # 直接序列化整个 rejected dict
                    rejected_content = json.dumps(rejected_data, ensure_ascii=False)
            elif isinstance(rejected_data, str):
                try:
                    rejected_obj = json.loads(rejected_data)
                    rejected_content = json.dumps(rejected_obj, ensure_ascii=False)
                except json.JSONDecodeError:
                    rejected_content = rejected_data
            else:
                rejected_content = str(rejected_data)
            
            # 构建标准 ORPO 格式
            output_data = {
                "messages": messages,
                "rejected_response": rejected_content
            }
            
            # 保留原始 meta 信息（可选）
            if "meta" in data:
                output_data["meta"] = data["meta"]
            
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    
    print(f"转换完成！输出文件：{output_file}")


# 更简单的版本：直接把 rejected 的 thinking 作为 rejected_response
def convert_orpo_format_simple(input_file, output_file):
    """
    简单版本：直接将 rejected 的 thinking 作为 rejected_response
    """
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行 JSON 解析失败: {e}")
                continue
            
            messages = list(data['messages'])
            
            # chosen 直接作为 assistant 回复
            chosen_str = data.get('chosen', '{}')
            messages.append({
                "role": "assistant",
                "content": chosen_str if isinstance(chosen_str, str) else json.dumps(chosen_str, ensure_ascii=False)
            })
            
            # rejected 的 thinking 作为被拒绝的回复
            rejected_data = data.get('rejected', {})
            if isinstance(rejected_data, dict) and 'thinking' in rejected_data:
                rejected_response = rejected_data['thinking']
            else:
                rejected_response = json.dumps(rejected_data, ensure_ascii=False)
            
            output_data = {
                "messages": messages,
                "rejected_response": rejected_response
            }
            
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    
    print(f"转换完成！输出文件：{output_file}")


if __name__ == "__main__":
    # 先用简单版本试试
    convert_orpo_format_simple(
        "/root/autodl-tmp/EvoAtomMem/data/dpo/atom_dpo.jsonl",
        "/root/autodl-tmp/EvoAtomMem/data/dpo/atom_dpo_standard.jsonl"
    )