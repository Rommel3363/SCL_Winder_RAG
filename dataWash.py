from difflib import SequenceMatcher
from typing import List, Tuple

def get_function_blocks(lines: List[str]) -> List[Tuple[int, List[str]]]:
    """
    提取代码中的函数块
    
    Args:
        lines: 代码行列表
    
    Returns:
        List of tuples containing (start_line_number, function_lines)
    """
    function_blocks = []
    current_block = []
    current_start = -1
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # 检测函数开始（包含 winder 的行）
        if "winder" in stripped_line.lower() and ("(" in stripped_line or "{" in stripped_line):
            if current_block:  # 如果已有块，先保存
                function_blocks.append((current_start, current_block))
                current_block = []
            current_start = i
            current_block = [line]
        # 如果已经在收集函数块
        elif current_block:
            current_block.append(line)
            # 检测函数结束（右花括号单独一行）
            if stripped_line == '}':
                function_blocks.append((current_start, current_block))
                current_block = []
                current_start = -1
    
    # 处理最后一个块
    if current_block:
        function_blocks.append((current_start, current_block))
    
    return function_blocks

def is_similar_winder_pair(block1: List[str], block2: List[str]) -> bool:
    """
    检查两个函数块是否为相似的top/bottom winder对
    
    Args:
        block1: 第一个函数块的行列表
        block2: 第二个函数块的行列表
    
    Returns:
        bool: 是否为相似的winder对
    """
    # 将块转换为单个字符串进行比较
    text1 = ''.join(block1).lower()
    text2 = ''.join(block2).lower()
    
    # 检查一个是top一个是bottom
    has_top = "top" in text1.lower()
    has_bottom = "bottom" in text2.lower()
    if not (has_top ^ has_bottom):  # 确保一个是top一个是bottom
        return False
    
    # 计算相似度
    similarity = SequenceMatcher(None, text1, text2).ratio()
    return similarity > 0.8  # 相似度阈值可以调整

def clean_scl_code(input_file: str, output_file: str) -> None:
    """
    清洗SCL代码
    """
    try:
        # 读取文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 获取所有函数块
        function_blocks = get_function_blocks(lines)
        
        # 标记要删除的行
        lines_to_delete = set()
        processed_blocks = set()
        
        # 检查相邻的函数块
        for i in range(len(function_blocks) - 1):
            if i in processed_blocks:
                continue
                
            current_start, current_block = function_blocks[i]
            next_start, next_block = function_blocks[i + 1]
            
            # 检查是否为相似的winder对
            if is_similar_winder_pair(current_block, next_block):
                # 确定哪个是top，哪个是bottom
                current_text = ''.join(current_block).lower()
                if "bottom" in current_text:
                    bottom_range = range(current_start, current_start + len(current_block))
                    lines_to_delete.update(bottom_range)
                else:
                    bottom_range = range(next_start, next_start + len(next_block))
                    lines_to_delete.update(bottom_range)
                processed_blocks.add(i)
                processed_blocks.add(i + 1)
        
        # 处理保留的行
        cleaned_lines = []
        for i, line in enumerate(lines):
            if i in lines_to_delete:
                continue
                
            # 去除行首尾的空白字符
            stripped_line = line.strip()
            
            # 跳过空行
            if not stripped_line:
                continue
            
            # 跳过region和end_region行
            if any(stripped_line.lower().startswith(prefix) for prefix in ['#region', '#endregion', 'region', 'end_region']):
                continue
                
            # 检查是否是只包含 // 的注释行
            if stripped_line.startswith('//'):
                comment_content = stripped_line[2:].strip()
                if not comment_content or comment_content.strip('!@#$%^&*()_+-=[]{}|;:,.<>?/~`').strip() == '':
                    continue
            
            # 删除"top"字符
            if "winder" in line.lower():
                line = line.replace("top", "").replace("Top", "").replace("TOP", "")
            
            # 保留有效行
            cleaned_lines.append(line)
        
        # 写入清洗后的内容
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
            
        print(f"处理完成！")
        print(f"原始行数: {len(lines)}")
        print(f"清洗后行数: {len(cleaned_lines)}")
        print(f"删除的winder函数对数量: {len(processed_blocks) // 2}")
        print(f"总删除行数: {len(lines) - len(cleaned_lines)}")
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

if __name__ == "__main__":
    input_file = "combined_scl_codes.txt"
    output_file = "cleaned_scl_codes.txt"
    clean_scl_code(input_file, output_file)