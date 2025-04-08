# -*- coding: utf-8 -*-
import pandas as pd
import warnings

# Bỏ qua (ignore) các cảnh báo loại PerformanceWarning từ Pandas
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import calendar
import numpy as np
import time  # Thêm để đo thời gian (tùy chọn)

# ==================== CELL 1: ĐỌC DỮ LIỆU ====================
print("🔄 Bắt đầu đọc dữ liệu Parquet...")
start_read_time = time.time()  # Đo thời gian đọc
parquet_folder = r"C:\Khue\TDN\data\processed"

try:
    all_files_in_folder = os.listdir(parquet_folder)
    parquet_files = [
        os.path.join(parquet_folder, f)
        for f in all_files_in_folder
        if f.startswith("sanluong_") and f.endswith(".parquet")
    ]

    if not parquet_files:
        print(f"⚠️ Không tìm thấy file Parquet trong: {parquet_folder}")
        df_all = pd.DataFrame()
    else:
        df_list = []
        print(f"🔍 Tìm thấy {len(parquet_files)} file Parquet. Bắt đầu đọc...")
        for i, f in enumerate(parquet_files):
            try:
                df_temp = pd.read_parquet(f, engine="pyarrow")
                # Chỉ đọc các cột thực sự cần thiết ngay từ đầu để tiết kiệm bộ nhớ
                required_cols_read = ["CTDL", "NMTD", "MADIEMDO", "ENDTIME", "CS"]
                if all(col in df_temp.columns for col in required_cols_read):
                    df_list.append(df_temp[required_cols_read])  # Chỉ lấy cột cần thiết
                else:
                    print(f"   ⚠️ File {os.path.basename(f)} thiếu cột, bỏ qua.")
            except Exception as e:
                print(f"❌ Lỗi đọc file {os.path.basename(f)}: {e}")

        if df_list:
            print("   Ghép các DataFrame...")
            df_all = pd.concat(df_list, ignore_index=True)
            print(f"✅ Đọc và ghép {len(df_list)} file thành công.")
            print(f"👉 Tổng số dòng: {df_all.shape[0]:,}")
            print(f"⏱️ Thời gian đọc và ghép: {time.time() - start_read_time:.2f} giây")
            # Hiển thị thông tin bộ nhớ (tùy chọn)
            # df_all.info(memory_usage='deep')
        else:
            print("❌ Không đọc được file nào thành công.")
            df_all = pd.DataFrame()

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy thư mục: {parquet_folder}")
    df_all = pd.DataFrame()
except Exception as e:
    print(f"❌ Lỗi không xác định khi đọc file/thư mục: {e}")
    df_all = pd.DataFrame()


# ==================== CELL 2: TIỀN XỬ LÝ VÀ TỐI ƯU HÓA ====================
df_sanluong = pd.DataFrame()
df_sanluong_indexed = pd.DataFrame()

if not df_all.empty:
    print("\n🔄 Bắt đầu tiền xử lý và tối ưu hóa...")
    start_process_time = time.time()
    try:
        # 1. Đổi tên cột và đảm bảo kiểu dữ liệu datetime
        df_sanluong = df_all.rename(
            columns={"ENDTIME": "TIME"}, errors="raise"
        )  # Đảm bảo cột tồn tại
        df_sanluong["TIME"] = pd.to_datetime(df_sanluong["TIME"])

        # 2. Đảm bảo kiểu dữ liệu số cho CS và xử lý NaN
        df_sanluong["CS"] = pd.to_numeric(df_sanluong["CS"], errors="coerce")
        initial_rows = len(df_sanluong)
        df_sanluong.dropna(subset=["CS"], inplace=True)
        removed_rows = initial_rows - len(df_sanluong)
        if removed_rows > 0:
            print(f"   ⚠️ Đã loại bỏ {removed_rows:,} dòng có giá trị CS không hợp lệ.")

        # 3. Tạo các cột thời gian cần thiết
        df_sanluong["YEAR"] = df_sanluong["TIME"].dt.year.astype(
            "int16"
        )  # Tối ưu kiểu int
        df_sanluong["MONTH_NUM"] = df_sanluong["TIME"].dt.month.astype("int8")
        df_sanluong["DAY_NUM"] = df_sanluong["TIME"].dt.day.astype("int8")
        df_sanluong["TIME_SLOT"] = df_sanluong["TIME"].dt.strftime("%H:%M")

        # 4. Tối ưu kiểu dữ liệu Categorical cho các cột chuỗi lặp lại
        df_sanluong["CTDL"] = df_sanluong["CTDL"].astype("category")
        df_sanluong["NMTD"] = df_sanluong["NMTD"].astype("category")
        df_sanluong["MADIEMDO"] = df_sanluong["MADIEMDO"].astype("category")
        df_sanluong["TIME_SLOT"] = df_sanluong["TIME_SLOT"].astype("category")

        # df_sanluong bây giờ chứa tất cả dữ liệu cần thiết với kiểu tối ưu
        print("   ✅ DataFrame gốc đã được xử lý và tối ưu hóa kiểu dữ liệu.")
        # df_sanluong.info(memory_usage='deep') # Kiểm tra lại bộ nhớ

        # 5. Tạo DataFrame được Index cho tính toán thống kê nhanh
        index_cols = ["CTDL", "NMTD", "MADIEMDO", "YEAR", "MONTH_NUM", "DAY_NUM"]
        # Chỉ cần TIME_SLOT và CS cho df_indexed
        df_sanluong_indexed = df_sanluong[index_cols + ["TIME_SLOT", "CS"]].copy()
        # Giữ kiểu category cho index để tiết kiệm bộ nhớ index
        df_sanluong_indexed.set_index(index_cols, inplace=True)
        df_sanluong_indexed.sort_index(inplace=True)  # Rất quan trọng cho hiệu năng loc
        print("   ✅ DataFrame index cho thống kê đã được tạo và sắp xếp.")
        # df_sanluong_indexed.info(memory_usage='deep')

        print(f"⏱️ Thời gian tiền xử lý: {time.time() - start_process_time:.2f} giây")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình tiền xử lý: {e}")
        # Đảm bảo cả hai df đều rỗng nếu có lỗi
        df_sanluong = pd.DataFrame()
        df_sanluong_indexed = pd.DataFrame()
else:
    print("ℹ️ Không có dữ liệu đầu vào để xử lý.")

# ==================== CELL 3: GIAO DIỆN TƯƠNG TÁC VÀ VẼ BIỂU ĐỒ ====================

# --- Chỉ tiếp tục nếu dữ liệu đã được xử lý thành công ---
if not df_sanluong.empty and not df_sanluong_indexed.empty:
    print("\n🔄 Tạo giao diện tương tác...")
    # ========== DANH SÁCH CHO DROPDOWN (Lấy từ categories để nhanh hơn) ==========
    try:
        ctdl_list = sorted(df_sanluong["CTDL"].cat.categories)
        year_list = sorted(df_sanluong["YEAR"].unique())  # Year là số, dùng unique
        # Lấy danh sách NMTD ban đầu (tất cả) - sẽ được lọc sau
        nmtd_list_all = sorted(df_sanluong["NMTD"].cat.categories)
    except Exception as e:
        print(f"❌ Lỗi lấy danh sách dropdown: {e}")
        ctdl_list, year_list, nmtd_list_all = [], [], []

    # ========== HÀM TẠO DROPDOWN ==========
    def make_dropdown(options, description, width="250px", margin="0px 20px 0px 0px"):
        safe_options = list(options) if options is not None else []
        # Thêm giá trị None vào đầu nếu muốn có lựa chọn trống (tùy chọn)
        # safe_options = [(None, '--Chọn--')] + safe_options
        return widgets.Dropdown(
            options=safe_options,
            description=description,
            value=safe_options[0] if safe_options else None,  # Chọn giá trị đầu tiên
            layout=widgets.Layout(width=width, margin=margin),
            style={"description_width": "auto"},
            disabled=not bool(safe_options),
        )

    # ========== DROPDOWNS ==========
    # (Tạo dropdown như cũ, sử dụng ctdl_list, year_list)
    # Tab 1
    ctdl_dropdown_month_plot = make_dropdown(ctdl_list, "CTDL:", "300px")
    nmtd_dropdown_month_plot = make_dropdown([], "Nhà máy:", "300px")  # Bắt đầu rỗng
    year_dropdown_month_plot = make_dropdown(year_list, "Năm:", "120px")
    month_dropdown_month_plot = make_dropdown(list(range(1, 13)), "Tháng:", "120px")
    placeholder_month_plot = widgets.Label(
        value="", layout=widgets.Layout(width="120px", margin="0px 20px 0px 0px")
    )
    # Tab 2
    ctdl_dropdown_day_plot = make_dropdown(ctdl_list, "CTDL:", "300px")
    nmtd_dropdown_day_plot = make_dropdown([], "Nhà máy:", "300px")
    year_dropdown_day_plot = make_dropdown(year_list, "Năm:", "120px")
    month_dropdown_day_plot = make_dropdown(list(range(1, 13)), "Tháng:", "120px")
    day_dropdown_day_plot = make_dropdown([], "Ngày:", "120px")
    # Tab 3
    ctdl_dropdown_stat = make_dropdown(ctdl_list, "CTDL:", "300px")
    nmtd_dropdown_stat = make_dropdown([], "Nhà máy:", "300px")
    madiemdo_dropdown_stat = make_dropdown([], "Mã điểm đo:", "300px")
    year_dropdown_stat = make_dropdown(year_list, "Năm:", "120px")
    month_dropdown_stat = make_dropdown(list(range(1, 13)), "Tháng:", "120px")

    # ========== NHÓM WIDGET ==========
    month_plot_widgets = [
        ctdl_dropdown_month_plot,
        nmtd_dropdown_month_plot,
        year_dropdown_month_plot,
        month_dropdown_month_plot,
    ]
    day_plot_widgets = [
        ctdl_dropdown_day_plot,
        nmtd_dropdown_day_plot,
        year_dropdown_day_plot,
        month_dropdown_day_plot,
        day_dropdown_day_plot,
    ]
    stat_widgets = [
        ctdl_dropdown_stat,
        nmtd_dropdown_stat,
        madiemdo_dropdown_stat,
        year_dropdown_stat,
        month_dropdown_stat,
    ]

    # ========== CẬP NHẬT NHÀ MÁY (Tối ưu: Lọc trên index category) ==========
    # Cache để tránh lọc lại NMTD cho cùng CTDL nhiều lần
    nmtd_cache = {}
    madiemdo_cache = {}

    def update_nmtd(ctdl_value, dropdown_to_update):
        global nmtd_cache
        options = []
        current_nmtd_value = dropdown_to_update.value

        if ctdl_value:
            if ctdl_value in nmtd_cache:  # Kiểm tra cache trước
                options = nmtd_cache[ctdl_value]
            elif (
                not df_sanluong_indexed.empty
            ):  # Chỉ lọc nếu chưa có trong cache và df tồn tại
                try:
                    # Lấy NMTD categories từ index con tương ứng với CTDL
                    # Điều này nhanh hơn .loc[ctdl_value].index... trên df lớn
                    nmtd_options = (
                        df_sanluong_indexed.loc[ctdl_value]
                        .index.get_level_values("NMTD")
                        .unique()
                        .tolist()
                    )
                    options = sorted(nmtd_options)
                    nmtd_cache[ctdl_value] = options  # Lưu vào cache
                except KeyError:
                    options = []  # CTDL không có trong index
                except Exception as e:
                    print(f"Lỗi cập nhật NMTD cache cho '{ctdl_value}': {e}")
                    options = []

        # Cập nhật dropdown như cũ
        dropdown_to_update.options = options
        dropdown_to_update.disabled = not bool(options)
        if current_nmtd_value in options:
            dropdown_to_update.value = current_nmtd_value
        elif options:
            dropdown_to_update.value = options[0]
        else:
            dropdown_to_update.value = None

    def update_madiemdo(ctdl_value, nmtd_value, dropdown_to_update):
        global madiemdo_cache
        options = []
        current_madiemdo_value = dropdown_to_update.value

        if ctdl_value and nmtd_value:
            cache_key = f"{ctdl_value}_{nmtd_value}"
            if cache_key in madiemdo_cache:  # Kiểm tra cache trước
                options = madiemdo_cache[cache_key]
            elif not df_sanluong.empty:  # Chỉ lọc nếu chưa có trong cache
                try:
                    # Lấy danh sách các mã điểm đo của nhà máy đã chọn
                    madiemdo_options = (
                        df_sanluong[
                            (df_sanluong["CTDL"] == ctdl_value)
                            & (df_sanluong["NMTD"] == nmtd_value)
                        ]["MADIEMDO"]
                        .unique()
                        .tolist()
                    )

                    options = sorted(madiemdo_options)
                    madiemdo_cache[cache_key] = options  # Lưu vào cache
                except Exception as e:
                    print(f"Lỗi cập nhật MADIEMDO cache cho '{cache_key}': {e}")
                    options = []

        # Cập nhật dropdown
        dropdown_to_update.options = options
        dropdown_to_update.disabled = not bool(options)
        if current_madiemdo_value in options:
            dropdown_to_update.value = current_madiemdo_value
        elif options:
            dropdown_to_update.value = options[0]
        else:
            dropdown_to_update.value = None

    # Gắn observe và khởi tạo NMTD (như cũ)
    ctdl_dropdown_month_plot.observe(
        lambda change: update_nmtd(change["new"], nmtd_dropdown_month_plot),
        names="value",
    )
    ctdl_dropdown_day_plot.observe(
        lambda change: update_nmtd(change["new"], nmtd_dropdown_day_plot), names="value"
    )
    ctdl_dropdown_stat.observe(
        lambda change: update_nmtd(change["new"], nmtd_dropdown_stat), names="value"
    )
    nmtd_dropdown_stat.observe(
        lambda change: update_madiemdo(
            ctdl_dropdown_stat.value, change["new"], madiemdo_dropdown_stat
        ),
        names="value",
    )
    # Khởi tạo
    if ctdl_dropdown_month_plot.value:
        update_nmtd(ctdl_dropdown_month_plot.value, nmtd_dropdown_month_plot)
    if ctdl_dropdown_day_plot.value:
        update_nmtd(ctdl_dropdown_day_plot.value, nmtd_dropdown_day_plot)
    if ctdl_dropdown_stat.value:
        update_nmtd(ctdl_dropdown_stat.value, nmtd_dropdown_stat)
    if ctdl_dropdown_stat.value and nmtd_dropdown_stat.value:
        update_madiemdo(
            ctdl_dropdown_stat.value, nmtd_dropdown_stat.value, madiemdo_dropdown_stat
        )

    # ========== GIỚI HẠN NGÀY (Như cũ) ==========
    def update_day_options(*args):
        year = year_dropdown_day_plot.value
        month = month_dropdown_day_plot.value
        current_day = day_dropdown_day_plot.value
        day_options = []
        if year and month:
            try:
                max_day = calendar.monthrange(year, month)[1]
                day_options = list(range(1, max_day + 1))
            except Exception as e:
                print(f"Lỗi lấy số ngày {month}/{year}: {e}")
        day_dropdown_day_plot.options = day_options
        day_dropdown_day_plot.disabled = not bool(day_options)
        if current_day in day_options:
            day_dropdown_day_plot.value = current_day
        elif day_options:
            day_dropdown_day_plot.value = 1
        else:
            day_dropdown_day_plot.value = None

    year_dropdown_day_plot.observe(update_day_options, names="value")
    month_dropdown_day_plot.observe(update_day_options, names="value")
    if year_dropdown_day_plot.value and month_dropdown_day_plot.value:
        update_day_options()

    # ========== VÙNG OUTPUT CHUNG ==========
    out = widgets.Output()

    # ========== VẼ BIỂU ĐỒ THEO MADIEMDO (Tab 1 & 2 - Tối ưu lọc) ==========
    def plot_filtered(mode, ctdl, nmtd, year, month, day=None):
        with out:
            clear_output(wait=True)
            plot_start_time = time.time()

            # Kiểm tra đầu vào
            required_vals = [ctdl, nmtd, year, month]
            if mode == "day":
                required_vals.append(day)
            if not all(v is not None for v in required_vals):
                print("   ⚠️ Vui lòng chọn đủ các bộ lọc.")
                return

            filtered_df = pd.DataFrame()
            try:
                # *** Tối ưu lọc: Lọc trên df_sanluong đã tối ưu kiểu dữ liệu ***
                # Tạo mask boolean hiệu quả
                mask = (
                    (df_sanluong["CTDL"] == ctdl)
                    & (df_sanluong["NMTD"] == nmtd)
                    & (df_sanluong["YEAR"] == year)
                    & (df_sanluong["MONTH_NUM"] == month)
                )
                if mode == "day":
                    mask &= df_sanluong["DAY_NUM"] == day

                # Chỉ lấy các cột cần thiết cho vẽ (`TIME`, `CS`, `MADIEMDO`) sau khi lọc
                filtered_df = df_sanluong.loc[mask, ["TIME", "CS", "MADIEMDO"]].copy()
                # Sắp xếp theo TIME là cần thiết cho biểu đồ đường
                filtered_df.sort_values("TIME", inplace=True)

            except Exception as e:
                print(f"❌ Lỗi khi lọc dữ liệu cho biểu đồ: {e}")

            if filtered_df.empty:
                print("   ")
                return

            # --- Vẽ biểu đồ (Giữ cấu hình gốc, thêm webgl) ---
            try:
                title_ext = (
                    f"{month}/{year}" if mode == "month" else f"{day}/{month}/{year}"
                )
                fig = px.line(
                    filtered_df,
                    x="TIME",
                    y="CS",
                    color="MADIEMDO",
                    render_mode="webgl",  # Quan trọng cho nhiều điểm
                    markers=True if mode == "day" else False,
                )
                # Áp dụng layout gốc
                fig.update_layout(
                    title=None,
                    annotations=[
                        dict(
                            text=f"<b>Công suất theo chu kỳ 30' - {nmtd} - {title_ext}</b>",
                            xref="paper",
                            yref="paper",
                            x=0.45,
                            y=1.1,
                            xanchor="center",
                            yanchor="top",
                            showarrow=False,
                            font=dict(size=18, color="black"),
                        )
                    ],
                    template="plotly_white",
                    height=600,
                    hovermode="x unified",
                    xaxis_title=None,
                    yaxis_title="Giá trị công suất",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="lightgrey",
                        griddash="dot",
                        title_font=dict(size=14, color="black"),
                        ticklabelposition="outside",
                        ticks="outside",
                        ticklen=8,
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="lightgrey",
                        griddash="dot",
                        rangemode="tozero",
                        title_font=dict(size=14, color="black", weight="bold"),
                        ticklabelposition="outside",
                        ticks="outside",
                        ticklen=8,
                    ),
                    margin=dict(l=60, r=40, t=80, b=80),
                    paper_bgcolor="white",
                    plot_bgcolor="#f0f8ff",
                )
                fig.add_annotation(
                    text="<b>Thời gian</b>",
                    xref="paper",
                    yref="paper",
                    x=0.45,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                )
                fig.update_traces(
                    line=dict(width=2),
                    hovertemplate="Công suất: %{y}",
                )

                fig.show()

            except Exception as e:
                print(f"❌ Lỗi khi vẽ biểu đồ Plotly Express: {e}")

    # ========== VẼ BIỂU ĐỒ THỐNG KÊ THEO KHUNG GIỜ (Tab 3 - Tối ưu lọc) ==========
    def plot_timeslot_stats(ctdl, nmtd, madiemdo, year, month):
        with out:
            clear_output(wait=True)

            if not all([ctdl, nmtd, year, month]):
                print("   ⚠️ Vui lòng chọn đủ CTDL, Nhà máy, Năm và Tháng.")
                return

            stats_by_time = pd.DataFrame()
            try:
                # *** Tối ưu lọc: Dùng .loc trên df_sanluong_indexed ***
                idx_query = (ctdl, nmtd, madiemdo, year, month)
                # Kiểm tra sự tồn tại của index hiệu quả hơn
                if idx_query in df_sanluong_indexed.index.droplevel("DAY_NUM").unique():
                    # .loc trực tiếp trên index đã sắp xếp là rất nhanh
                    monthly_data = df_sanluong_indexed.loc[
                        idx_query, ["TIME_SLOT", "CS"]
                    ]

                    if not monthly_data.empty:
                        # Group và Agg - bước này thường nhanh trên dữ liệu đã lọc
                        stats_by_time = monthly_data.groupby(
                            "TIME_SLOT", observed=True
                        )["CS"].agg(  # observed=True tăng tốc nếu TIME_SLOT là category
                            min="min",
                            p25=lambda x: x.quantile(0.25),
                            p50="mean",
                            p75=lambda x: x.quantile(0.75),
                            max="max",
                        )
                        # Sắp xếp index (TIME_SLOT) nếu chưa đúng thứ tự (thường thì groupby giữ nguyên)
                        stats_by_time = stats_by_time.sort_index()
                    else:
                        print("   ℹ️ Không có dữ liệu chi tiết sau khi lọc.")

            except KeyError:
                print("   ℹ️ Không tìm thấy dữ liệu (KeyError).")
            except Exception as e:
                print(f"❌ Lỗi tính toán thống kê: {e}")

            if stats_by_time.empty:
                print("   ℹ️ Không có dữ liệu cho lựa chọn này!")
                return

            # --- Vẽ biểu đồ (Giữ nguyên cấu hình Plotly GO) ---
            try:
                fig = go.Figure()
                time_slots = stats_by_time.index
                p25 = stats_by_time["p25"]
                p75 = stats_by_time["p75"]
                p50 = stats_by_time["p50"]
                min_cs = stats_by_time["min"]
                max_cs = stats_by_time["max"]

                # 1. Vùng Min-Max (Màu xám nhạt giống ảnh)
                fig.add_trace(
                    go.Scatter(
                        x=time_slots,
                        y=max_cs,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",  # Không hiện trong legend, không hover
                        hovertemplate="Max: %{y:.2f}<extra></extra>",  # Bỏ nếu không cần hover max
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_slots,
                        y=min_cs,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(211, 211, 211, 0.5)",  # Light Gray
                        name="Min-Max",  # Tên trong legend
                        hoverinfo="skip",  # Chỉ hiển thị hover ở trace dưới nếu muốn
                        hovertemplate="Min: %{y:.2f}<extra></extra>",  # Bỏ nếu không cần hover min
                    )
                )

                # 2. Vùng P25-P75 (Màu xanh nhạt giống ảnh)
                fig.add_trace(
                    go.Scatter(
                        x=time_slots,
                        y=p75,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        hovertemplate="P75: %{y:.2f}<extra></extra>",  # Bỏ nếu không cần hover P75
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_slots,
                        y=p25,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(173, 216, 230, 0.6)",  # Light Blue
                        name="Tần suất 25-75",  # Tên trong legend
                        hoverinfo="skip",
                        hovertemplate="P25: %{y:.2f}<extra></extra>",  # Bỏ nếu không cần hover P25
                        # hovertemplate='Khung giờ: %{x}<br>P25: %{y:.2f}<extra></extra>'
                    )
                )

                # 3. Đường Median (P50) (Màu xanh dương đậm)
                fig.add_trace(
                    go.Scatter(
                        x=time_slots,
                        y=p50,
                        mode="lines+markers",
                        marker=dict(size=6, color="blue"),  # Màu xanh dương đậm
                        line=dict(color="blue", width=2.5),  # Dày hơn chút
                        name="Trung bình",  # Tên trong legend
                        hovertemplate="Mean: %{y:.2f}<extra></extra>",
                    )
                )

                # Layout
                fig.update_layout(
                    title=None,
                    annotations=[
                        dict(
                            text=f"<b>Thống kê công suất theo khung giờ - {nmtd} - {month}/{year}</b>",
                            xref="paper",
                            yref="paper",
                            x=0.45,
                            y=1.1,
                            xanchor="center",
                            yanchor="top",
                            showarrow=False,
                            font=dict(size=18, color="black"),
                        )
                    ],
                    template="plotly_white",
                    height=600,
                    hovermode="x unified",
                    xaxis_title=None,
                    yaxis_title="Giá trị công suất",
                    xaxis=dict(
                        tickmode="array",
                        tickvals=time_slots[:],  # Tất cả các khung giờ
                        ticktext=[t for t in time_slots],
                        showgrid=True,
                        gridcolor="lightgrey",
                        griddash="dot",
                        title_font=dict(size=14, color="black"),
                        ticklabelposition="outside",
                        ticks="outside",
                        ticklen=8,
                        tickangle=-90,  # Xoay nhãn nếu cần
                    ),
                    yaxis=dict(
                        rangemode="tozero",
                        showgrid=True,
                        gridcolor="lightgrey",
                        griddash="dot",
                        title_font=dict(size=14, color="black", weight="bold"),
                        ticklabelposition="outside",
                        ticks="outside",
                        ticklen=8,
                    ),
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    margin=dict(l=60, r=40, t=80, b=80),
                    paper_bgcolor="white",
                    plot_bgcolor="#f0f8ff",
                )
                fig.add_annotation(
                    text="<b>Khung giờ</b>",
                    xref="paper",
                    yref="paper",
                    x=0.45,
                    y=-0.18,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                )
                fig.show()

            except Exception as e:
                print(f"❌ Lỗi vẽ biểu đồ thống kê Plotly GO: {e}")

    # ========== HÀM XỬ LÝ SỰ KIỆN THAY ĐỔI DROPDOWN (Giữ nguyên) ==========
    def on_change_month_plot(change):
        if tabs.selected_index == 0 and all(
            w.value is not None for w in month_plot_widgets
        ):
            if change is None or (
                isinstance(change, dict) and change.get("new") != change.get("old")
            ):
                plot_filtered(
                    "month",
                    ctdl_dropdown_month_plot.value,
                    nmtd_dropdown_month_plot.value,
                    year_dropdown_month_plot.value,
                    month_dropdown_month_plot.value,
                )

    def on_change_day_plot(change):
        if tabs.selected_index == 1 and all(
            w.value is not None for w in day_plot_widgets
        ):
            if change is None or (
                isinstance(change, dict) and change.get("new") != change.get("old")
            ):
                plot_filtered(
                    "day",
                    ctdl_dropdown_day_plot.value,
                    nmtd_dropdown_day_plot.value,
                    year_dropdown_day_plot.value,
                    month_dropdown_day_plot.value,
                    day_dropdown_day_plot.value,
                )

    def on_change_stat(change):
        if tabs.selected_index == 2 and all(w.value is not None for w in stat_widgets):
            if change is None or (
                isinstance(change, dict) and change.get("new") != change.get("old")
            ):
                plot_timeslot_stats(
                    ctdl_dropdown_stat.value,
                    nmtd_dropdown_stat.value,
                    madiemdo_dropdown_stat.value,
                    year_dropdown_stat.value,
                    month_dropdown_stat.value,
                )

    # ========== HÀM TẮT / BẬT OBSERVER (Giữ nguyên) ==========
    _observed_callbacks = {}

    def disable_observe(widgets_list):
        for w in widgets_list:
            if w in _observed_callbacks and _observed_callbacks[w]["active"]:
                try:
                    w.unobserve(_observed_callbacks[w]["func"], names="value")
                    _observed_callbacks[w]["active"] = False
                except:
                    _observed_callbacks[w]["active"] = (
                        False  # Ignore errors during unobserve
                    )

    def enable_observe(widgets_list):
        for w in widgets_list:
            if (
                w in _observed_callbacks
                and not w.disabled
                and not _observed_callbacks[w]["active"]
            ):
                try:
                    w.observe(_observed_callbacks[w]["func"], names="value")
                    _observed_callbacks[w]["active"] = True
                except Exception as e:
                    print(f"Lỗi observe {w.description}: {e}")

    # ========== GẮN OBSERVER BAN ĐẦU (Giữ nguyên) ==========
    for w in month_plot_widgets:
        _observed_callbacks[w] = {"func": on_change_month_plot, "active": False}
    for w in day_plot_widgets:
        _observed_callbacks[w] = {"func": on_change_day_plot, "active": False}
    for w in stat_widgets:
        _observed_callbacks[w] = {"func": on_change_stat, "active": False}

    # ========== TẠO GIAO DIỆN TABS (Giữ nguyên) ==========
    tab_month_plot = widgets.HBox(
        month_plot_widgets + [placeholder_month_plot],
        layout=widgets.Layout(
            justify_content="flex-start", align_items="flex-end", overflow="visible"
        ),
    )
    tab_day_plot = widgets.HBox(
        day_plot_widgets,
        layout=widgets.Layout(
            justify_content="flex-start", align_items="flex-end", overflow="visible"
        ),
    )
    tab_stat = widgets.HBox(
        stat_widgets,
        layout=widgets.Layout(
            justify_content="flex-start", align_items="flex-end", overflow="visible"
        ),
    )
    tabs = widgets.Tab(children=[tab_month_plot, tab_day_plot, tab_stat])
    tabs.set_title(0, "📅 Biểu đồ tháng")
    tabs.set_title(1, "🗓️ Biểu đồ ngày")
    tabs.set_title(2, "📊 Thống kê tháng")

    # ========== XỬ LÝ CHUYỂN TAB (Đồng bộ hóa - Sửa lại current_tab_index) ==========
    # Sử dụng biến này trong scope của cell, không cần global
    current_tab_index = tabs.selected_index

    # ========== XỬ LÝ CHUYỂN TAB (Đồng bộ hóa - Sửa lại) ==========
    # Không cần biến current_tab_index nữa vì ta sẽ sử dụng change["old"]
    def on_tab_change(change):
        if change["name"] == "selected_index":
            new_tab_index = change["new"]
            old_tab_index = change["old"]  # Sử dụng giá trị old từ thay đổi

            # --- Tắt observer tab cũ ---
            if old_tab_index == 0:
                disable_observe(month_plot_widgets)
            elif old_tab_index == 1:
                disable_observe(day_plot_widgets)
            elif old_tab_index == 2:
                disable_observe(stat_widgets)

            # --- Xác định nguồn đồng bộ ---
            source_widgets = {}  # Dictionary để lưu widget nguồn
            if old_tab_index == 0:
                source_widgets = {
                    "ctdl": ctdl_dropdown_month_plot,
                    "nmtd": nmtd_dropdown_month_plot,
                    "year": year_dropdown_month_plot,
                    "month": month_dropdown_month_plot,
                }
            elif old_tab_index == 1:
                source_widgets = {
                    "ctdl": ctdl_dropdown_day_plot,
                    "nmtd": nmtd_dropdown_day_plot,
                    "year": year_dropdown_day_plot,
                    "month": month_dropdown_day_plot,
                    "day": day_dropdown_day_plot,
                }
            elif old_tab_index == 2:
                source_widgets = {
                    "ctdl": ctdl_dropdown_stat,
                    "nmtd": nmtd_dropdown_stat,
                    "year": year_dropdown_stat,
                    "month": month_dropdown_stat,
                }
            else:  # Khởi tạo
                source_widgets = {
                    "ctdl": ctdl_dropdown_month_plot,
                    "nmtd": nmtd_dropdown_month_plot,
                    "year": year_dropdown_month_plot,
                    "month": month_dropdown_month_plot,
                }

            # --- Xác định widgets đích của tab mới ---
            target_widgets = {}
            if new_tab_index == 0:
                target_widgets = {
                    "ctdl": ctdl_dropdown_month_plot,
                    "nmtd": nmtd_dropdown_month_plot,
                    "year": year_dropdown_month_plot,
                    "month": month_dropdown_month_plot,
                }
            elif new_tab_index == 1:
                target_widgets = {
                    "ctdl": ctdl_dropdown_day_plot,
                    "nmtd": nmtd_dropdown_day_plot,
                    "year": year_dropdown_day_plot,
                    "month": month_dropdown_day_plot,
                    "day": day_dropdown_day_plot,
                }
            elif new_tab_index == 2:
                target_widgets = {
                    "ctdl": ctdl_dropdown_stat,
                    "nmtd": nmtd_dropdown_stat,
                    "madiemdo": madiemdo_dropdown_stat,
                    "year": year_dropdown_stat,
                    "month": month_dropdown_stat,
                }

            # --- Đồng bộ từ nguồn sang đích ---
            if old_tab_index != new_tab_index and old_tab_index is not None:
                # Đồng bộ CTDL
                if (
                    source_widgets.get("ctdl")
                    and source_widgets["ctdl"].value is not None
                ):
                    target_widgets["ctdl"].value = source_widgets["ctdl"].value

                # Đồng bộ NMTD (phải cập nhật options trước)
                target_nmtd = target_widgets["nmtd"]
                update_nmtd(target_widgets["ctdl"].value, target_nmtd)
                if (
                    source_widgets.get("nmtd")
                    and source_widgets["nmtd"].value in target_nmtd.options
                ):
                    target_nmtd.value = source_widgets["nmtd"].value
                # Đồng bộ MADIEMDO (phải cập nhật options trước)
                if new_tab_index == 2 and "madiemdo" in target_widgets:
                    update_madiemdo(
                        target_widgets["ctdl"].value,
                        target_widgets["nmtd"].value,
                        target_widgets["madiemdo"],
                    )

                # Đồng bộ Year
                if (
                    source_widgets.get("year")
                    and source_widgets["year"].value is not None
                ):
                    target_widgets["year"].value = source_widgets["year"].value

                # Đồng bộ Month
                if (
                    source_widgets.get("month")
                    and source_widgets["month"].value is not None
                ):
                    target_widgets["month"].value = source_widgets["month"].value

                # Nếu chuyển sang tab ngày, cập nhật dropdown ngày
                if "day" in target_widgets:
                    update_day_options()  # Cập nhật options ngày dựa trên năm/tháng đã chọn
                    # Mặc định chọn ngày 1 nếu đến từ tab khác
                    if old_tab_index != 1 and target_widgets["day"].options:
                        target_widgets["day"].value = 1

            # --- Kích hoạt observer và cập nhật nội dung tab mới ---
            if new_tab_index == 0:
                enable_observe(month_plot_widgets)
                on_change_month_plot(None)  # Gọi hàm vẽ/tính toán tương ứng
            elif new_tab_index == 1:
                enable_observe(day_plot_widgets)
                on_change_day_plot(None)
            elif new_tab_index == 2:
                enable_observe(stat_widgets)
                on_change_stat(None)

    tabs.observe(on_tab_change, names="selected_index")

    # ========== HIỂN THỊ ==========
    print("✅ Giao diện sẵn sàng!")
    display(tabs, out)

    # ========== KÍCH HOẠT BAN ĐẦU ==========
    # Kích hoạt lần đầu bằng cách gọi on_tab_change mô phỏng với old=None
    on_tab_change(
        {
            "name": "selected_index",
            "old": None,
            "new": tabs.selected_index,
            "owner": tabs,
            "type": "change",
        }
    )
elif df_all.empty:
    print("\n❌ Không có dữ liệu đầu vào từ file Parquet.")
else:  # df_sanluong hoặc df_sanluong_indexed rỗng
    print("\n❌ Lỗi tiền xử lý hoặc không có dữ liệu hợp lệ.")
