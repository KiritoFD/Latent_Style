import subprocess
import psutil
import os
import json

def get_mem_hardware():
    # 使用 PowerShell 获取物理内存详情，输出为 JSON 格式便于解析
    cmd = 'powershell "Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed | ConvertTo-Json"'
    try:
        res = subprocess.check_output(cmd, shell=True).decode('utf-8')
        data = json.loads(res)
        # 兼容单条内存和多条内存的返回格式
        items = data if isinstance(data, list) else [data]
        return [int(item['Capacity']) // (1024**3) for item in items]
    except Exception:
        # 如果获取失败，let it crash，但在 crash 前给个提示
        print("Hardware detection failed. Make sure PowerShell is available.")
        raise

def run_diagnosis():
    slots = get_mem_hardware()
    total_physical = sum(slots)
    
    # Flex Mode 逻辑：对称区域是最小条目的两倍
    dual_zone_size = min(slots) * 2
    
    vm = psutil.virtual_memory()
    used_gb = vm.used / (1024**3)
    
    print(f"--- 7940HX Memory Topology ---")
    print(f"Slots Layout:  {slots} GB")
    print(f"Fast Zone:     0 - {dual_zone_size} GB (Dual Channel)")
    print(f"Slow Zone:     {dual_zone_size} - {total_physical} GB (Single Channel)")
    
    print(f"\n--- Real-time Status ---")
    print(f"System Used:   {used_gb:.2f} GB")
    
    if used_gb > dual_zone_size:
        print(f"Status: !!! CRITICAL - Data is spilling into Single Channel zone.")
    else:
        margin = dual_zone_size - used_gb
        print(f"Status: Optimal - {margin:.2f} GB headroom left in Dual Channel.")

    # 针对你 AI 项目的建议
    print(f"\n--- Infra Suggestion ---")
    if total_physical < 48: # 预防识别错误
        print("Check hardware connection.")
    else:
        print(f"Ensure your VAE/Latent datasets stay within the first {dual_zone_size}GB.")

if __name__ == "__main__":
    run_diagnosis()