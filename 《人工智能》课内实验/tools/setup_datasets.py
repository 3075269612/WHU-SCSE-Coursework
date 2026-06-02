"""
数据集准备工具 — 为第6章实验准备所需的外部数据集
支持自动下载（优先）和本地生成模拟数据（备用）
运行方式：在项目根目录执行  python tools/setup_datasets.py
"""
import os, sys, ssl, zipfile, shutil, urllib.request
import numpy as np

DATA_DIR = 'data'

# ============================================================
# 工具函数
# ============================================================
def download(url, dest):
    """下载文件（兼容 SSL 受限环境）"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            with open(dest, 'wb') as f:
                f.write(resp.read())

# ============================================================
# 1. secom.data
# ============================================================
def setup_secom():
    fp = os.path.join(DATA_DIR, 'secom.data')
    print("[1/3] SECOM 半导体数据 (data/secom.data)")
    if os.path.exists(fp):
        print("  -> 已存在，跳过")
        return
    try:
        print("  -> 尝试从 UCI 下载...")
        download('https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data', fp)
        print("  -> 下载成功")
    except Exception as e:
        print(f"  -> 下载失败 ({e})，改由本地生成模拟数据...")
        _generate_secom(fp)

def _generate_secom(fp):
    np.random.seed(42)
    n_samples, n_features = 1567, 591
    print(f"  -> 生成 {n_samples}x{n_features} 矩阵...")
    data = np.random.randn(n_samples, n_features) * 2 + 10
    mask = np.random.random(data.shape) < 0.045
    data[mask] = np.nan
    with open(fp, 'w') as f:
        for row in data:
            f.write(' '.join(['NaN' if np.isnan(v) else f'{v:.6f}' for v in row]) + '\n')
    print("  -> 模拟数据生成完成")

# ============================================================
# 2. ORL 人脸数据集
# ============================================================
def setup_orl():
    orl_dir = os.path.join(DATA_DIR, 'ORL')
    print("[2/3] ORL 人脸数据集 (data/ORL/)")
    if os.path.exists(orl_dir) and len(os.listdir(orl_dir)) >= 40:
        print("  -> 已存在，跳过")
        return
    try:
        print("  -> 尝试从 Cambridge 下载 AT&T 人脸数据库...")
        download('https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip',
                 '_att_faces.zip')
        print("  -> 解压中...")
        with zipfile.ZipFile('_att_faces.zip', 'r') as zf:
            zf.extractall('.')
        os.remove('_att_faces.zip')
        os.makedirs(orl_dir, exist_ok=True)
        # 查找 s1-s40 目录
        skip = {'data', 'docs', 'output', 'tools', '__pycache__',
                '.claude', '.venv', '.vscode'}
        moved = 0
        for d in sorted(os.listdir('.')):
            path = os.path.join('.', d)
            if not os.path.isdir(path) or d in skip or d.startswith('.'):
                continue
            subs = [s for s in os.listdir(path)
                    if s.startswith('s') and s[1:].isdigit()]
            if subs:
                for s in subs:
                    shutil.move(os.path.join(path, s), os.path.join(orl_dir, s))
                    moved += 1
                shutil.rmtree(path)
                break
            if d.startswith('s') and d[1:].isdigit():
                shutil.move(path, os.path.join(orl_dir, d))
                moved += 1
        # 残余
        if moved < 40:
            for d in sorted(os.listdir('.')):
                path = os.path.join('.', d)
                if os.path.isdir(path) and d.startswith('s') and d[1:].isdigit():
                    try:
                        shutil.move(path, os.path.join(orl_dir, d))
                        moved += 1
                    except Exception:
                        pass
        # pgm → jpg
        renamed = 0
        for root, _, files in os.walk(orl_dir):
            for fn in files:
                if fn.endswith('.pgm'):
                    os.rename(os.path.join(root, fn),
                              os.path.join(root, fn.replace('.pgm', '.jpg')))
                    renamed += 1
        print(f"  -> 下载并解压完成 (移动 {moved} 子目录, 重命名 {renamed} 文件)")
    except Exception as e:
        print(f"  -> 下载失败 ({e})，改由本地生成模拟数据...")
        _generate_orl(orl_dir)

def _generate_orl(orl_dir):
    """PGM P5 二进制格式，无需 cv2/PIL，cv2.imread 可读"""
    np.random.seed(123)
    os.makedirs(orl_dir, exist_ok=True)
    x = np.linspace(0, 4 * np.pi, 92)
    y = np.linspace(0, 4 * np.pi, 112)
    X, Y = np.meshgrid(x, y)
    for person in range(1, 41):
        person_dir = os.path.join(orl_dir, f's{person}')
        os.makedirs(person_dir, exist_ok=True)
        fx, fy = 1 + (person % 7), 1 + (person % 5)
        phase = person * 0.3
        base = np.sin(fx * X + phase) * np.cos(fy * Y + phase)
        base = ((base + 1) / 2 * 255).astype(np.uint8)
        for img in range(1, 11):
            noise = np.random.randint(-25, 25, (112, 92)).astype(np.uint8)
            face = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
            header = f'P5\n92 112\n255\n'.encode('ascii')
            with open(os.path.join(person_dir, f'{img}.jpg'), 'wb') as f:
                f.write(header + face.tobytes())
        if person % 10 == 0:
            print(f"  -> 已生成 {person}/40 人")
    print("  -> 模拟 ORL 数据生成完成")

# ============================================================
# 3. testSet.txt
# ============================================================
def setup_testset():
    fp = os.path.join(DATA_DIR, 'testSet.txt')
    print("[3/3] testSet.txt (data/testSet.txt)")
    if os.path.exists(fp):
        print("  -> 已存在，跳过")
        return
    np.random.seed(123)
    centers = [(2.0, 2.0), (-2.0, -2.0), (2.5, -2.0), (-2.5, 2.5)]
    data = np.vstack([np.random.randn(20, 2) * 0.5 + [cx, cy] for cx, cy in centers])
    with open(fp, 'w') as f:
        for row in data:
            f.write(f'{row[0]:.6f}\t{row[1]:.6f}\n')
    print("  -> 生成完成 (80 个点, 4 个聚类)")

# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"工作目录: {os.getcwd()}\n")
    setup_secom()
    setup_orl()
    setup_testset()
    print("\n全部数据集准备完成！可以运行第6章脚本了。")
