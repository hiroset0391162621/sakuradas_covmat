"""
specw_data/sakura2022/ 以下の npz ファイルを期間全体で読み込み，
overview の spectrogram plot を PDF として出力するスクリプト．

出力: figure/specw_sakura2022/specw_overview_{start}_{end}.pdf
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import datetime

try:
    import scienceplots
except ImportError:
    pass

plt.style.use(["science", "nature"])
plt.rcParams['xtick.direction'] = "inout"
plt.rcParams['ytick.direction'] = "inout"
plt.rcParams["text.usetex"] = False
plt.rcParams['agg.path.chunksize'] = 100000
plt.rcParams["date.converter"] = "concise"
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
SPECW_DIR   = "specw_data/sakura2022"
FIGURE_DIR  = "figure/specw_sakura2022"
LOW_PASS    = 0.1   # [Hz] y 軸下限
HIGH_PASS   = 10.0  # [Hz] y 軸上限
VMIN        = None  # pcolormesh の clim（None で自動）
VMAX        = None


def load_all_npz(specw_dir: str) -> tuple:
    """specw_dir 内の npz ファイルを時刻順に全て読み込み結合して返す．

    Returns
    -------
    datetime_num : np.ndarray  shape (Nt,)
        matplotlib date number（全期間を時刻順に連結）
    frequencies  : np.ndarray  shape (Nf+1,)
        周波数軸（最後のファイルの値を代表として使用）
    spectral_width : np.ndarray  shape (Nf, Nt)
        結合後の spectral width
    """
    files = sorted(glob.glob(os.path.join(specw_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"npz ファイルが見つかりません: {specw_dir}")

    datetime_num_list   = []
    spectral_width_list = []
    frequencies = None

    for f in files:
        data = np.load(f)
        datetime_num_list.append(data["datetime_num"])
        spectral_width_list.append(data["spectral_width"])  # shape: (Nf, Nt_i)
        frequencies = data["frequencies"]
        print(f"読み込み完了: {os.path.basename(f)}  "
              f"t={data['datetime_num'].shape}  "
              f"sw={data['spectral_width'].shape}")

    datetime_num   = np.concatenate(datetime_num_list, axis=0)
    spectral_width = np.concatenate(spectral_width_list, axis=1)  # 時刻方向に連結

    return datetime_num, frequencies, spectral_width


def make_output_filename(datetime_num: np.ndarray, figure_dir: str) -> str:
    """開始・終了日付から PDF ファイル名を生成する．"""
    t_start = mdates.num2date(datetime_num[0]).strftime("%Y%m%d")
    t_end   = mdates.num2date(datetime_num[-1]).strftime("%Y%m%d")
    os.makedirs(figure_dir, exist_ok=True)
    return os.path.join(figure_dir, f"specw_overview_{t_start}_{t_end}.pdf")


def plot_overview(datetime_num, frequencies, spectral_width, output_path):
    """overview スペクトログラムを 1 ページの PDF として保存する．"""

    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)

    # pcolormesh 用のグリッド
    # datetime_num は各タイムウィンドウの左端（または中心）の matplotlib date number
    # frequencies は savez 時点で linspace(ymin, ymax, Nf+1) で保存されている
    Nf, Nt = spectral_width.shape

    # x 方向：均等ステップで bin 境界を作成
    if Nt > 1:
        dx = datetime_num[1] - datetime_num[0]
    else:
        dx = 0.0
    x_edges = np.append(datetime_num, datetime_num[-1] + dx)

    X, Y = np.meshgrid(x_edges, frequencies)

    img = ax.pcolormesh(
        X, Y, spectral_width,
        cmap="RdYlBu",
        shading="flat",
        rasterized=True,
        vmin=VMIN,
        vmax=VMAX,
    )

    cbar = plt.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("spectral width", fontsize=11)

    ax.set_ylim(LOW_PASS, HIGH_PASS)
    ax.set_yscale("log")
    ax.set_ylabel("Frequency [Hz]", fontsize=12)

    # x 軸を日付表示
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", length=4, width=1)
    ax.tick_params(axis="both", which="minor", length=2, width=0.75)
    ax.tick_params(which="both", direction="out")

    t_start_str = mdates.num2date(datetime_num[0]).strftime("%Y-%m-%d")
    t_end_str   = mdates.num2date(datetime_num[-1]).strftime("%Y-%m-%d")
    ax.set_title(f"Spectral width overview  {t_start_str} – {t_end_str}", fontsize=12)

    # with PdfPages(output_path) as pdf:
    #     pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.show(fig)
    print(f"保存完了: {output_path}")


if __name__ == "__main__":
    datetime_num, frequencies, spectral_width = load_all_npz(SPECW_DIR)
    output_path = make_output_filename(datetime_num, FIGURE_DIR)
    plot_overview(datetime_num, frequencies, spectral_width, output_path)
