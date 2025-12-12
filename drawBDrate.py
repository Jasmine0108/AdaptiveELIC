# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_rd_curve(dataset_path, output_filename='rd_curve.png'):
#     """
#     遍歷資料集路徑，計算每個設定的平均 BPP 和 PSNR，並繪製 RD Curve。
#     """
    
#     # 用來儲存每個設定點的列表 (BPP, PSNR, SettingName)
#     data_points = []
    
#     print(f"正在掃描資料集路徑: {dataset_path} ...")

#     # 1. 遍歷 dataset_path 下的所有資料夾
#     # os.listdir 列出所有子資料夾，我們假設這些是像 '0004', '0008' 的設定
#     subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
#     # 為了顯示順序好看，我們先對資料夾名稱排序 (非必要，因為畫圖會根據 BPP 排序，但 log 比較好看)
#     subdirs.sort() 

#     for folder_name in subdirs:
#         # 建構 result.csv 的完整路徑
#         # 結構: dataset_path / folder_name / results / result.csv
#         csv_path = os.path.join(dataset_path, folder_name, 'results', 'result.csv')
        
#         if os.path.exists(csv_path):
#             try:
#                 # 2. 讀取 CSV
#                 df = pd.read_csv(csv_path)
                
#                 # 檢查必要的欄位是否存在
#                 if 'bpp' in df.columns and 'psnr(dB)' in df.columns:
#                     # 3. 計算平均值
#                     avg_bpp = df['bpp'].mean()
#                     avg_psnr = df['psnr(dB)'].mean()
                    
#                     # 也可以計算 MS-SSIM，如果有需要的話
#                     # avg_msssim = df['ms-ssim'].mean()

#                     print(f"設定 [{folder_name}]: 平均 BPP = {avg_bpp:.4f}, 平均 PSNR = {avg_psnr:.4f}")
                    
#                     data_points.append({
#                         'bpp': avg_bpp,
#                         'psnr': avg_psnr,
#                         'setting': folder_name
#                     })
#                 else:
#                     print(f"警告: 檔案 {csv_path} 缺少 'bpp' 或 'psnr(dB)' 欄位，已跳過。")
            
#             except Exception as e:
#                 print(f"讀取 {csv_path} 時發生錯誤: {e}")
#         else:
#             # 如果某個資料夾沒有 results/result.csv 則單純跳過 (例如 .ipynb_checkpoints)
#             continue

#     if not data_points:
#         print("未找到任何有效的數據點，請檢查路徑結構。")
#         return

#     # 4. 整理數據以進行繪圖
#     # 將列表轉換為 DataFrame 方便處理
#     df_plot = pd.DataFrame(data_points)
    
#     # **關鍵步驟**: 依照 BPP 由小到大排序，這樣畫出來的線才不會亂跑
#     df_plot = df_plot.sort_values(by='bpp')

#     # 5. 開始繪圖
#     plt.figure(figsize=(10, 6))
    
#     # 繪製線條與點
#     plt.plot(df_plot['bpp'], df_plot['psnr'], marker='o', linestyle='-', color='b', label='Method Result')
    
#     # 在每個點旁邊標註設定名稱 (例如 0004, 0008)，方便除錯分析
#     for i, txt in enumerate(df_plot['setting']):
#         plt.annotate(txt, (df_plot['bpp'].iloc[i], df_plot['psnr'].iloc[i]), 
#                      xytext=(5, -10), textcoords='offset points', fontsize=8, color='gray')

#     # 設定圖表標籤
#     plt.title(f'{os.path.basename(dataset_path)}', fontsize=14)
#     plt.xlabel('Bitrate (bpp)', fontsize=12)
#     plt.ylabel('PSNR (dB)', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
    
#     # 儲存圖片
#     plt.savefig(output_filename)
#     print(f"\n圖表已儲存至: {output_filename}")
    
#     # 顯示圖表 (如果在 Jupyter Notebook 環境下)
#     # plt.show()

# # ==========================================
# # 使用範例
# # ==========================================

# # 請將這裡改成你實際的資料集路徑
# # 例如: r"C:\Users\Name\Project\craters" 或相對路徑 "craters"
# dataset_name = str(input("input dataset name: "))
# input_dataset_path = f"./results/{dataset_name}" 

# plot_rd_curve(input_dataset_path, output_filename=f"./{dataset_name}_bd_rate_curve.png")

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_curve_data_from_folders(dataset_path):
    """
    邏輯 A (你的方法): 遍歷資料夾 (0004, 0008...) -> 讀取 results/result.csv
    回傳: DataFrame (含 bpp, psnr)
    """
    data_points = []
    if not os.path.exists(dataset_path):
        print(f"找不到方法路徑: {dataset_path}")
        return pd.DataFrame()

    print(f"正在掃描方法資料集: {dataset_path} ...")
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    subdirs.sort() 

    for folder_name in subdirs:
        csv_path = os.path.join(dataset_path, folder_name, 'results', 'result.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'bpp' in df.columns and 'psnr(dB)' in df.columns:
                    data_points.append({
                        'bpp': df['bpp'].mean(),
                        'psnr': df['psnr(dB)'].mean(),
                        'setting': folder_name
                    })
            except Exception as e:
                print(f"讀取 {csv_path} 錯誤: {e}")
    
    # 轉成 DataFrame 並排序
    df_res = pd.DataFrame(data_points)
    if not df_res.empty:
        df_res = df_res.sort_values(by='bpp')
    return df_res

def get_curve_data_from_files(baseline_folder, dataset_name):
    """
    邏輯 B (Baseline): 根據 dataset_name 搜尋特定檔案 -> 讀取 CSV
    格式範例: Craters_ELIC_0004_ft_3980_Plateau_inference.csv
    """
    data_points = []
    
    # 搜尋模式: 資料夾路徑 + Dataset開頭 + _ELIC_ + 結尾是.csv
    # 例如: .../Craters_ELIC_*.csv
    search_pattern = os.path.join(baseline_folder, f"{dataset_name}_ELIC_*.csv")
    files = glob.glob(search_pattern)
    
    print(f"正在掃描 Baseline ({len(files)} 個檔案 found in {baseline_folder})...")

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            if 'bpp' in df.columns and 'psnr(dB)' in df.columns:
                data_points.append({
                    'bpp': df['bpp'].mean(),
                    'psnr': df['psnr(dB)'].mean(),
                    'filename': os.path.basename(file_path)
                })
        except Exception as e:
            print(f"讀取 Baseline {file_path} 錯誤: {e}")

    # 轉成 DataFrame 並排序
    df_res = pd.DataFrame(data_points)
    if not df_res.empty:
        df_res = df_res.sort_values(by='bpp')
    return df_res

def plot_comparison_rd_curve(method_df, baseline_df, dataset_name, output_filename):
    plt.figure(figsize=(10, 6))
    
    # 1. 畫 Baseline (通常用虛線或不同顏色表示對照組)
    if not baseline_df.empty:
        plt.plot(baseline_df['bpp'], baseline_df['psnr'], 
                 marker='^', linestyle='-', color='orange', label='ELIC Baseline', alpha=0.8)
    else:
        print("警告: 沒有 Baseline 數據可畫。")

    # 2. 畫你的 Method (實線)
    if not method_df.empty:
        plt.plot(method_df['bpp'], method_df['psnr'], 
                 marker='o', linestyle='-', color='blue', label='Ours')
    else:
        print("警告: 沒有 Method 數據可畫。")

    # 設定圖表
    plt.title(f'{dataset_name}', fontsize=14)
    plt.xlabel('Bitrate (bpp)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 儲存
    plt.savefig(output_filename)
    print(f"\n比較圖表已儲存至: {output_filename}")
    # plt.show()

# ==========================================
# 主程式執行區
# ==========================================

# 1. 設定路徑
# 請確保這裡指向包含 csv 的資料夾 (例如你提供的 D:\final_project\...\ELIC_baseline)
# 注意：Windows路徑建議在字串前加 r，或是用 / 取代 \
BASELINE_DIR = r"D:\final_project\image_compression\ELIC_baseline" 

dataset_name = str(input("input dataset name (e.g., Craters): "))
input_dataset_path = f"./results/{dataset_name}" 

# 2. 獲取數據
# 獲取你的方法數據
df_method = get_curve_data_from_folders(input_dataset_path)

# 獲取 Baseline 數據 (會自動搜尋 Craters_ELIC_*.csv)
df_baseline = get_curve_data_from_files(BASELINE_DIR, dataset_name)

# 3. 畫圖
if df_method.empty and df_baseline.empty:
    print("兩邊都沒有數據，無法繪圖。")
else:
    output_png = f"./{dataset_name}_comparison_rd_curve.png"
    plot_comparison_rd_curve(df_method, df_baseline, dataset_name, output_png)