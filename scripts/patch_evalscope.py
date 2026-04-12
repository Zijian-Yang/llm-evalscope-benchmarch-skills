#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
patch_evalscope.py - EvalScope空响应补丁

自动定位并修补EvalScope的空响应误判bug。

用法:
    python3.12 patch_evalscope.py [--venv-path ./venv]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


# 补丁标识
PATCH_MARKER = "# Empty response check (added by patch_evalscope.py)"

# 流式响应的补丁代码
STREAM_PATCH = '''
            # Empty response check (added by patch_evalscope.py)
            if not response_messages or not generated_text.strip():
                output.success = False
'''

# 非流式响应的补丁代码
NON_STREAM_PATCH = '''
        # Empty response check (added by patch_evalscope.py)
        if not response_messages or not generated_text.strip():
            output.success = False
'''


def find_evalscope_default_api(venv_path: Optional[Path] = None) -> Optional[Path]:
    """
    查找 evalscope 的 default_api.py 文件
    
    搜索顺序:
    1. 指定的 venv 路径
    2. 当前目录下的 venv
    3. 系统 site-packages
    """
    search_paths = []
    
    # 添加 venv 路径
    if venv_path:
        venv_path = Path(venv_path)
        # macOS/Linux
        search_paths.append(venv_path / "lib" / "python3.12" / "site-packages")
        # Windows
        search_paths.append(venv_path / "Lib" / "site-packages")
    
    # 当前目录的 venv
    cwd_venv = Path.cwd() / "venv"
    if cwd_venv.exists():
        search_paths.append(cwd_venv / "lib" / "python3.12" / "site-packages")
        search_paths.append(cwd_venv / "Lib" / "site-packages")
    
    # 系统 site-packages
    try:
        import site
        search_paths.extend([Path(p) for p in site.getsitepackages()])
        if site.getusersitepackages():
            search_paths.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    
    # 搜索 default_api.py
    target_file = "evalscope/perf/plugin/api/default_api.py"
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        target_path = base_path / target_file
        if target_path.exists():
            return target_path
    
    return None


def check_patch_status(file_path: Path) -> Tuple[bool, str]:
    """
    检查文件是否已打补丁
    
    返回: (是否已打补丁, 状态信息)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        
        if PATCH_MARKER in content:
            return True, "已存在补丁"
        
        return False, "未打补丁"
    
    except Exception as e:
        return False, f"读取失败: {e}"


def apply_patch(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    应用补丁
    
    返回: (是否成功, 结果信息)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        
        # 检查是否已打补丁
        if PATCH_MARKER in content:
            return True, "已存在补丁，无需重复打补丁"
        
        modified = False
        
        # 查找流式响应处理的位置
        # 通常在 "for chunk in response:" 循环结束后
        # 或在处理完 stream 数据后
        
        # 模式1: 查找流式处理结束的位置
        # 特征: 在 stream 处理循环后，通常有 output.stream = True 或类似标记
        stream_pattern = r'(output\.stream\s*=\s*True.*?\n)([\t ]*)(output\.success\s*=\s*True)'
        stream_match = re.search(stream_pattern, content, re.DOTALL)
        
        if stream_match:
            # 在 output.success = True 之前插入检查
            indent = stream_match.group(2)
            patch_code = f'\n{indent}{PATCH_MARKER}\n{indent}if not response_messages or not generated_text.strip():\n{indent}    output.success = False\n{indent}'
            content = content[:stream_match.end(1)] + patch_code + content[stream_match.start(3):]
            modified = True
        
        # 模式2: 查找非流式处理的位置
        # 特征: response = await ... 或 response = client.chat.completions.create(...)
        # 然后是 generated_text = ... 和 output.success = True
        non_stream_pattern = r'(generated_text\s*=.*?response.*?\n)([\t ]*)(output\.success\s*=\s*True)'
        non_stream_match = re.search(non_stream_pattern, content, re.DOTALL)
        
        if non_stream_match and PATCH_MARKER not in content:
            # 在 output.success = True 之前插入检查
            indent = non_stream_match.group(2)
            patch_code = f'\n{indent}{PATCH_MARKER}\n{indent}if not generated_text or not generated_text.strip():\n{indent}    output.success = False\n{indent}'
            content = content[:non_stream_match.end(1)] + patch_code + content[non_stream_match.start(3):]
            modified = True
        
        # 如果上述模式都没匹配，尝试更通用的模式
        if not modified:
            # 查找所有 output.success = True 的位置
            success_pattern = r'(\n)([\t ]+)(output\.success\s*=\s*True)'
            
            def replace_func(match):
                indent = match.group(2)
                patch = f'\n{indent}{PATCH_MARKER}\n{indent}if hasattr(output, "generated_text") and not getattr(output, "generated_text", "").strip():\n{indent}    output.success = False\n'
                return match.group(1) + patch + match.group(2) + match.group(3)
            
            new_content, count = re.subn(success_pattern, replace_func, content, count=1)
            
            if count > 0:
                content = new_content
                modified = True
        
        if not modified:
            return False, "未找到合适的补丁位置，请手动检查 default_api.py"
        
        if dry_run:
            return True, "补丁检查通过 (dry-run 模式，未实际应用)"
        
        # 备份原文件
        backup_path = file_path.with_suffix(".py.bak")
        backup_path.write_text(original_content, encoding="utf-8")
        
        # 写入修改后的内容
        file_path.write_text(content, encoding="utf-8")
        
        return True, f"补丁应用成功，原文件已备份到: {backup_path}"
    
    except Exception as e:
        return False, f"应用补丁失败: {e}"


def remove_patch(file_path: Path) -> Tuple[bool, str]:
    """
    移除补丁
    
    返回: (是否成功, 结果信息)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        
        if PATCH_MARKER not in content:
            return True, "文件中没有补丁"
        
        # 移除补丁代码块
        # 补丁通常是连续的几行，以 PATCH_MARKER 开始
        lines = content.split('\n')
        new_lines = []
        skip_until_dedent = False
        patch_indent = 0
        
        for line in lines:
            if PATCH_MARKER in line:
                skip_until_dedent = True
                patch_indent = len(line) - len(line.lstrip())
                continue
            
            if skip_until_dedent:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent <= patch_indent:
                    skip_until_dedent = False
                    new_lines.append(line)
                continue
            
            new_lines.append(line)
        
        new_content = '\n'.join(new_lines)
        
        # 备份并写入
        backup_path = file_path.with_suffix(".py.patched.bak")
        file_path.with_suffix(".py.patched.bak").write_text(content, encoding="utf-8")
        file_path.write_text(new_content, encoding="utf-8")
        
        return True, f"补丁已移除，打补丁版本备份到: {backup_path}"
    
    except Exception as e:
        return False, f"移除补丁失败: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="自动定位并修补EvalScope的空响应误判bug",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
此脚本用于修复 EvalScope 将空响应误判为成功的问题。

补丁逻辑：在流式/非流式响应处理完成后，检查 response_messages 和 
generated_text 是否为空，若为空则标记 output.success = False。

示例:
    # 自动查找并打补丁
    python3.12 patch_evalscope.py
    
    # 指定 venv 路径
    python3.12 patch_evalscope.py --venv-path ./venv
    
    # 只检查状态
    python3.12 patch_evalscope.py --check
    
    # 移除补丁
    python3.12 patch_evalscope.py --remove
        """
    )
    
    parser.add_argument(
        "--venv-path",
        type=str,
        default=None,
        help="虚拟环境路径 (默认自动查找)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查补丁状态，不修改文件"
    )
    
    parser.add_argument(
        "--remove",
        action="store_true",
        help="移除已应用的补丁"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟执行，不实际修改文件"
    )
    
    args = parser.parse_args()
    
    print("EvalScope 空响应补丁工具")
    print("=" * 40)
    
    # 查找 default_api.py
    venv_path = Path(args.venv_path) if args.venv_path else None
    file_path = find_evalscope_default_api(venv_path)
    
    if not file_path:
        print("错误: 未找到 evalscope 的 default_api.py", file=sys.stderr)
        print("\n可能的原因:")
        print("  1. evalscope 未安装")
        print("  2. venv 路径不正确")
        print("\n建议:")
        print("  pip install 'evalscope[perf]'")
        print("  或指定 --venv-path 参数")
        sys.exit(1)
    
    print(f"找到文件: {file_path}")
    
    # 检查状态
    is_patched, status = check_patch_status(file_path)
    print(f"当前状态: {status}")
    
    if args.check:
        sys.exit(0 if is_patched else 1)
    
    if args.remove:
        success, message = remove_patch(file_path)
        print(f"结果: {message}")
        sys.exit(0 if success else 1)
    
    # 应用补丁
    if is_patched:
        print("文件已有补丁，无需重复操作")
        sys.exit(0)
    
    success, message = apply_patch(file_path, dry_run=args.dry_run)
    print(f"结果: {message}")
    
    if success and not args.dry_run:
        print("\n补丁已应用。EvalScope 现在会正确处理空响应。")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
