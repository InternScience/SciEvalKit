import os
import json
import urllib.request
import sys


DATA_REPO_URL = "https://huggingface.co/datasets/JayKuo/EarthLink_files/resolve/main"

EXPECTED_FILES = [
    'avail_cmip6_data_mem_time.json',
    'avail_obs4mips_data.json',
    'avail_obs_data.json',
    'cmip6_activities.json',
    'cmip6_experiments.json',
    'cmip6_freqences.json',
    'cmip6_grid_labels.json',
    'cmip6_models.json',
    'cmip6_tables.json',
    'cmip6_variables.json',
    'derived_variables.json',
    'fx_variables.py',
    'obs4mips.py',
    'obs_data_info.json',
    'variables_embedding.jsonl'
]


def download_progress_hook(block_num, block_size, total_size):
    """显示下载进度的回调函数"""
    downloaded = block_num * block_size
    
    if total_size > 0:
        percent = min(downloaded * 100.0 / total_size, 100)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        # 创建进度条
        bar_length = 40
        filled_length = int(bar_length * downloaded / total_size)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 打印进度（使用 \r 回到行首实现覆盖效果）
        sys.stdout.write(f'\r  Progress: |{bar}| {percent:.1f}% ({downloaded_mb:.2f}/{total_mb:.2f} MB)')
        sys.stdout.flush()
        
        # 下载完成后换行
        if downloaded >= total_size:
            sys.stdout.write('\n')
            sys.stdout.flush()
    else:
        # 如果不知道总大小，只显示已下载的量
        downloaded_mb = downloaded / (1024 * 1024)
        sys.stdout.write(f'\r  Downloaded: {downloaded_mb:.2f} MB')
        sys.stdout.flush()


for file in EXPECTED_FILES:
    file_path = os.path.join(os.path.dirname(__file__), file)
    if not os.path.exists(file_path):
        # download the file
        url = f"{DATA_REPO_URL}/{file}"
        print(f"Downloading {file}...")
        try:
            urllib.request.urlretrieve(url, file_path, reporthook=download_progress_hook)
            print(f"✓ Successfully downloaded {file}")
        except Exception as e:
            print(f"✗ Failed to download {file}: {e}")


from .obs4mips import OBS4MIPS_MODELS
from .fx_variables import FX_VARIABLES


with open(os.path.join(os.path.dirname(__file__), 'obs_data_info.json'), 'r', encoding='utf-8') as f:
    OBS_DATA_INFO: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'cmip6_variables.json'), 'r', encoding='utf-8') as f:
    CMIP6_VARIABLES: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'derived_variables.json'), 'r', encoding='utf-8') as f:
    DERIVED_VARIABLES: dict = json.load(f)
for v in ['amoc', 'ohc', 'vegfrac']: # not supported currently
    DERIVED_VARIABLES.pop(v, None)  


with open(os.path.join(os.path.dirname(__file__), 'cmip6_models.json'), 'r', encoding='utf-8') as f:
    CMIP6_MODELS: dict = json.load(f)['source_id']


with open(os.path.join(os.path.dirname(__file__), 'cmip6_experiments.json'), 'r', encoding='utf-8') as f:
    CMIP6_EXPERIMENTS: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'cmip6_activities.json'), 'r', encoding='utf-8') as f:
    CMIP6_ACTIVITIES: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'cmip6_freqences.json'), 'r', encoding='utf-8') as f:
    CMIP6_FREQUENCES: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'cmip6_tables.json'), 'r', encoding='utf-8') as f:
    CMIP6_TABLES: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'avail_cmip6_data_mem_time.json'), 'r', encoding='utf-8') as f:
    AVAIL_CMIP6_DATA: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'avail_obs_data.json'), 'r', encoding='utf-8') as f:
    AVAIL_OBS_DATA: dict = json.load(f)


with open(os.path.join(os.path.dirname(__file__), 'avail_obs4mips_data.json'), 'r', encoding='utf-8') as f:
    AVAIL_OBS4MIPS_DATA: dict = json.load(f)
