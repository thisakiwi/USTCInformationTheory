import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 导入FineZip的核心类（确保eval_clean.py在Python路径中）
from fineziptiny import ZipModel

# ===================== 路径配置（按你的环境修改）=====================
FILE_NAME      = "enwik8_100KB"
DATA_DIR       = "./code/IT_Compression/data"
EVAL_CLEAN_PATH = "./code/fineziptiny.py" 

RAW_PATH       = os.path.join(DATA_DIR, f"{FILE_NAME}.txt")
COMP_PATH      = os.path.join('./code/data', f"{FILE_NAME}.ftz")  # FineZip输出的压缩文件
DECOMP_PATH    = os.path.join('./code/data', f"{FILE_NAME}.dec.txt")  # 解压后文件

# ===================== 初始化FineZip的ZipModel =====================
def init_finezip_model():
    # 匹配你的模型初始化参数
    CONTEXT_SIZE = 256
    BATCH_SIZE = 10
    model_path = "./code/TinyLlama"
    
    # 加载本地GPT2模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 初始化ZipModel
    zip_model = ZipModel(
        model_name=model_path,
        tokenizer_name=model_path,
        model=model,
        tokenizer=tokenizer,
        finetuned=False, 
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE
    )
    return zip_model

# ===================== 压缩 =====================
zip_model = init_finezip_model()
raw_size = os.path.getsize(RAW_PATH)

# 读取原始文本
with open(RAW_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 压缩计时
start_compress = time.perf_counter()
compressed_bytes = zip_model.encode_and_zip(raw_text)
# 保存压缩文件
with open(COMP_PATH, "wb") as f:
    f.write(compressed_bytes)
end_compress = time.perf_counter()
compress_time = end_compress - start_compress

# ===================== 解压 =====================
# 读取压缩文件
with open(COMP_PATH, "rb") as f:
    zipped_data = f.read()

# 解压计时
start_decompress = time.perf_counter()
decompressed_text = zip_model.unzip_and_decode(zipped_data)
# 保存解压文件
with open(DECOMP_PATH, "w", encoding="utf-8") as f:
    f.write(decompressed_text)
end_decompress = time.perf_counter()
decompress_time = end_decompress - start_decompress

# ===================== 计算指标 =====================
comp_size = os.path.getsize(COMP_PATH)
bpb = (comp_size * 8) / raw_size  # 每比特字节数
raw_mb = raw_size / (1024 * 1024)  # 原始大小(MB)

compress_throughput = raw_mb / compress_time  # 压缩吞吐量(MB/s)
decompress_throughput = raw_mb / decompress_time  # 解压吞吐量(MB/s)

# ===================== 输出结果 =====================
log_content = "tinyllama-FineZip ctx=256 压缩性能指标\n"
log_content += f"原始大小:{raw_size} B\n"
log_content += f"压缩后大小:{comp_size} B\n"
log_content += f"BPB:{bpb:.4f}\n"
log_content += f"压缩耗时:{compress_time:.6f} s\n"
log_content += f"解压耗时:{decompress_time:.6f} s\n"
log_content += f"压缩吞吐量:{compress_throughput:.2f} MB/s\n"
log_content += f"解压吞吐量:{decompress_throughput:.2f} MB/s\n"

# 打印到屏幕
print(log_content)

# 写入日志文件
with open("./code/fitizip_100KB.txt", "w", encoding="utf-8") as f:
    f.write(log_content)

# ===================== 无损校验 =====================
def compare_text_content(file1, file2):
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            text1 = f1.read().strip()
        with open(file2, 'r', encoding='utf-8') as f2:
            text2 = f2.read().strip()
        
        # 统一换行符（避免格式差异导致校验失败）
        text1 = text1.replace('\n', '').replace('\r', '')
        text2 = text2.replace('\n', '').replace('\r', '')
        
        return text1 == text2
    except Exception as e:
        print(f"校验出错: {e}")
        return False

is_same = compare_text_content(RAW_PATH, DECOMP_PATH)
check_msg = "无损校验通过！" if is_same else "解压文件与原文件不一致！"

print(check_msg)

# 追加校验结果到日志
with open("./code/fitizip_100KB.txt", "a", encoding="utf-8") as f:
    f.write(check_msg + "\n")