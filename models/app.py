import warnings
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go # Cần cho biểu đồ rỗng
import dash
from dash import dcc, html, Input, Output, State, no_update
import calendar

# Bỏ qua (ignore) các cảnh báo loại PerformanceWarning từ Pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ==================== 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU (Thực hiện một lần khi khởi chạy) ====================
print("🔄 Bắt đầu đọc và xử lý dữ liệu...")
parquet_folder = r"C:\Khue\TDN\data\processed" # <-- Đảm bảo đường dẫn này chính xác
df_sanluong = pd.DataFrame() # Khởi tạo DataFrame rỗng

try:
    all_files_in_folder = os.listdir(parquet_folder)
    parquet_files = [os.path.join(parquet_folder, f)
                     for f in all_files_in_folder
                     if f.startswith("sanluong_") and f.endswith(".parquet")]

    if not parquet_files:
        print(f"⚠️ Không tìm thấy file nào khớp mẫu 'sanluong_*.parquet' trong thư mục: {parquet_folder}")
    else:
        df_list = []
        for f in parquet_files:
            try:
                df_list.append(pd.read_parquet(f, engine='pyarrow'))
            except Exception as e:
                print(f"❌ Lỗi khi đọc file {f}: {e}")

        if df_list:
            df_all = pd.concat(df_list, ignore_index=True)
            print(f"✅ Đã đọc xong {len(parquet_files)} file parquet. Tổng số dòng ban đầu: {df_all.shape[0]:,}")

            # --- Xử lý df_sanluong ---
            required_cols = ['CTDL', 'NMTD', 'MADIEMDO', 'ENDTIME', 'CS']
            if all(col in df_all.columns for col in required_cols):
                df_sanluong = df_all[required_cols].copy()
                df_sanluong.rename(columns={'ENDTIME': 'TIME'}, inplace=True)

                # ++++++++++++++++ TỐI ƯU HÓA KIỂU DỮ LIỆU (BẮT ĐẦU) ++++++++++++++++
                if not df_sanluong.empty:
                    print("🔄 Tối ưu hóa kiểu dữ liệu DataFrame...")
                    mem_usage_before_opt = df_sanluong.memory_usage(deep=True).sum() / (1024**2)
                    print(f"   📊 Bộ nhớ sử dụng TRƯỚC khi tối ưu: {mem_usage_before_opt:.2f} MB")

                    # Chuyển đổi các cột ID/chuỗi lặp lại thành 'category'
                    for col in ['CTDL', 'NMTD', 'MADIEMDO']:
                        if col in df_sanluong.columns:
                            try:
                                df_sanluong[col] = df_sanluong[col].astype('category')
                                print(f"   ✅ Đã chuyển cột '{col}' sang kiểu 'category'.")
                            except Exception as e:
                                print(f"   ⚠️ Không thể chuyển cột '{col}' sang category: {e}")

                    # Chuyển đổi cột số công suất sang float32 nếu có thể
                    if 'CS' in df_sanluong.columns:
                        try:
                            df_sanluong['CS'] = pd.to_numeric(df_sanluong['CS'], errors='coerce') # Đảm bảo là số
                            # Chỉ downcast nếu không mất mát lớn (kiểm tra min/max nếu cần)
                            df_sanluong['CS'] = df_sanluong['CS'].astype('float32')
                            print("   ✅ Đã chuyển cột 'CS' sang kiểu 'float32'.")
                        except Exception as e:
                            print(f"   ⚠️ Không thể chuyển cột 'CS' sang float32: {e}")

                    mem_usage_after_opt = df_sanluong.memory_usage(deep=True).sum() / (1024**2)
                    print(f"   📊 Bộ nhớ sử dụng SAU khi tối ưu kiểu dữ liệu cơ bản: {mem_usage_after_opt:.2f} MB")
                # ++++++++++++++++ TỐI ƯU HÓA KIỂU DỮ LIỆU (KẾT THÚC) ++++++++++++++++


                # --- Tiếp tục xử lý thời gian và index ---
                print("🔄 Xử lý thời gian và tối ưu DataFrame...")
                try:
                    df_sanluong['TIME'] = pd.to_datetime(df_sanluong['TIME'])

                    # TẠO và TỐI ƯU kiểu dữ liệu cột thời gian NGAY LẬP TỨC
                    df_sanluong['YEAR'] = df_sanluong['TIME'].dt.year.astype('int16') # Năm thường đủ int16
                    df_sanluong['MONTH_NUM'] = df_sanluong['TIME'].dt.month.astype('int8') # Tháng đủ int8
                    df_sanluong['DAY_NUM'] = df_sanluong['TIME'].dt.day.astype('int8')    # Ngày đủ int8
                    print("   ✅ Đã tạo và tối ưu kiểu dữ liệu cột YEAR, MONTH_NUM, DAY_NUM.")

                    mem_usage_after_time = df_sanluong.memory_usage(deep=True).sum() / (1024**2)
                    print(f"   📊 Bộ nhớ sử dụng sau khi thêm cột thời gian tối ưu: {mem_usage_after_time:.2f} MB")


                    # --- Tiếp tục đặt index và sắp xếp ---
                    index_cols = ['CTDL', 'NMTD', 'YEAR', 'MONTH_NUM', 'DAY_NUM']
                    if all(col in df_sanluong.columns for col in index_cols):
                        print("   ⏳ Bắt đầu đặt index...")
                        df_sanluong.set_index(index_cols, inplace=True)
                        print("   ⏳ Bắt đầu sắp xếp index (có thể mất thời gian và bộ nhớ)...")
                        df_sanluong.sort_index(inplace=True) # Đây vẫn là bước tốn kém nhất
                        print("✅ Đã xử lý thời gian và tối ưu index DataFrame.")
                        mem_usage_final = df_sanluong.memory_usage(deep=True).sum() / (1024**2)
                        print(f"   📊 Bộ nhớ sử dụng cuối cùng: {mem_usage_final:.2f} MB")
                    else:
                        print("❌ Thiếu cột cần thiết để tạo index tối ưu.")
                        missing_idx_cols = [col for col in index_cols if col not in df_sanluong.columns]
                        print(f"   -> Các cột index bị thiếu: {missing_idx_cols}")
                        df_sanluong = pd.DataFrame() # Đặt lại thành rỗng nếu lỗi
                except Exception as e: # Bắt cả MemoryError và các lỗi khác
                    print(f"❌ Lỗi khi xử lý thời gian hoặc đặt index: {e}")
                    df_sanluong = pd.DataFrame() # Đặt lại thành rỗng nếu lỗi
            else:
                print(f"❌ Thiếu cột cần thiết trong dữ liệu ban đầu. Cần có: {required_cols}")
                df_sanluong = pd.DataFrame()
        else:
            print("❌ Không đọc được file nào thành công.")

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy thư mục: {parquet_folder}")
    df_sanluong = pd.DataFrame() # Đảm bảo df_sanluong rỗng nếu lỗi
except Exception as e:
    print(f"❌ Lỗi không xác định khi truy cập thư mục hoặc file: {e}")
    df_sanluong = pd.DataFrame() # Đảm bảo df_sanluong rỗng nếu lỗi


# --- Lấy danh sách ban đầu cho dropdowns (nếu df_sanluong tồn tại và không rỗng) ---
initial_ctdl_options = []
initial_year_options = []
if not df_sanluong.empty: # Kiểm tra lại df_sanluong sau xử lý
    try:
        # Kiểm tra xem index có tồn tại không trước khi truy cập
        if isinstance(df_sanluong.index, pd.MultiIndex):
             if 'CTDL' in df_sanluong.index.names:
                 initial_ctdl_options = [{'label': c, 'value': c} for c in sorted(df_sanluong.index.get_level_values('CTDL').unique())]
             if 'YEAR' in df_sanluong.index.names:
                 initial_year_options = [{'label': y, 'value': y} for y in sorted(df_sanluong.index.get_level_values('YEAR').unique())]
        else:
             print("⚠️ DataFrame không có MultiIndex như mong đợi sau khi xử lý.")

    except Exception as e:
        print(f"❌ Lỗi khi lấy danh sách CTDL/Year ban đầu từ index: {e}")

# ==================== 2. KHỞI TẠO ỨNG DỤNG DASH ====================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ==================== 3. ĐỊNH NGHĨA LAYOUT ỨNG DỤNG ====================
app.layout = html.Div([
    html.H1("Trực Quan Hóa Dữ Liệu Sản Lượng", style={'textAlign': 'center'}),

    dcc.Tabs(id="tabs-main", value='tab-month', children=[
        # --- Tab Xem Theo Tháng ---
        dcc.Tab(label='📅 Xem theo tháng', value='tab-month', children=[
            html.Div([
                html.Div([
                    html.Label("CTDL:", style={'margin-right': '5px'}),
                    dcc.Dropdown(
                        id='ctdl-dropdown-month',
                        options=initial_ctdl_options,
                        # Chọn giá trị đầu tiên nếu có, nếu không thì None
                        value=initial_ctdl_options[0]['value'] if initial_ctdl_options else None,
                        clearable=False,
                        style={'width': '300px', 'margin-right': '20px'}
                    )
                ], style={'display': 'flex', 'align-items': 'center'}),
                html.Div([
                    html.Label("Nhà máy:", style={'margin-right': '5px'}),
                    dcc.Dropdown(
                        id='nmtd-dropdown-month',
                        options=[], # Sẽ được cập nhật bởi callback
                        placeholder="Chọn nhà máy...",
                        clearable=False,
                        disabled=True, # Bắt đầu bị vô hiệu hóa
                        style={'width': '300px', 'margin-right': '20px'}
                    )
                ], style={'display': 'flex', 'align-items': 'center'}),
                html.Div([
                     html.Label("Năm:", style={'margin-right': '5px'}),
                     dcc.Dropdown(
                        id='year-dropdown-month',
                        options=initial_year_options,
                        value=initial_year_options[0]['value'] if initial_year_options else None,
                        clearable=False,
                        style={'width': '120px', 'margin-right': '20px'}
                     )
                ], style={'display': 'flex', 'align-items': 'center'}),
                html.Div([
                     html.Label("Tháng:", style={'margin-right': '5px'}),
                     dcc.Dropdown(
                        id='month-dropdown-month',
                        options=[{'label': m, 'value': m} for m in range(1, 13)],
                        value=1, # Giá trị tháng mặc định
                        clearable=False,
                        style={'width': '120px'}
                    )
                ], style={'display': 'flex', 'align-items': 'center'}),
            ], style={'display': 'flex', 'padding': '20px', 'flexWrap': 'wrap'}) # flexWrap để xuống dòng trên màn hình nhỏ
        ]),

        # --- Tab Xem Theo Ngày ---
        dcc.Tab(label='🗓️ Xem theo ngày', value='tab-day', children=[
            html.Div([
                 html.Div([
                    html.Label("CTDL:", style={'margin-right': '5px'}),
                    dcc.Dropdown(
                        id='ctdl-dropdown-day',
                        options=initial_ctdl_options,
                        value=initial_ctdl_options[0]['value'] if initial_ctdl_options else None,
                        clearable=False,
                        style={'width': '300px', 'margin-right': '20px'}
                    )
                ], style={'display': 'flex', 'align-items': 'center'}),
                html.Div([
                    html.Label("Nhà máy:", style={'margin-right': '5px'}),
                    dcc.Dropdown(
                        id='nmtd-dropdown-day',
                        options=[], # Sẽ được cập nhật bởi callback
                        placeholder="Chọn nhà máy...",
                        clearable=False,
                        disabled=True,
                        style={'width': '300px', 'margin-right': '20px'}
                    )
                ], style={'display': 'flex', 'align-items': 'center'}),
                 html.Div([
                     html.Label("Năm:", style={'margin-right': '5px'}),
                     dcc.Dropdown(
                        id='year-dropdown-day',
                        options=initial_year_options,
                        value=initial_year_options[0]['value'] if initial_year_options else None,
                        clearable=False,
                        style={'width': '120px', 'margin-right': '20px'}
                     )
                 ], style={'display': 'flex', 'align-items': 'center'}),
                html.Div([
                     html.Label("Tháng:", style={'margin-right': '5px'}),
                     dcc.Dropdown(
                        id='month-dropdown-day',
                        options=[{'label': m, 'value': m} for m in range(1, 13)],
                        value=1,
                        clearable=False,
                        style={'width': '120px', 'margin-right': '20px'}
                    )
                ], style={'display': 'flex', 'align-items': 'center'}),
                html.Div([
                     html.Label("Ngày:", style={'margin-right': '5px'}),
                     dcc.Dropdown(
                        id='day-dropdown-day',
                        options=[], # Sẽ được cập nhật bởi callback
                        placeholder="Chọn ngày...",
                        clearable=False,
                        disabled=True,
                        style={'width': '120px'}
                    )
                ], style={'display': 'flex', 'align-items': 'center'}),
            ], style={'display': 'flex', 'padding': '20px', 'flexWrap': 'wrap'})
        ]),
    ]),

    # --- Khu vực hiển thị biểu đồ và thông báo ---
    html.Div(id='status-message', style={'padding': '10px', 'font-style': 'italic', 'color': 'grey'}),
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=dcc.Graph(id='graph-output', figure=go.Figure()) # Bắt đầu với biểu đồ trống
    )
])

# ==================== 4. ĐỊNH NGHĨA CALLBACKS ====================
# --- Đồng bộ filter khi chuyển sang tab ngày ---
@app.callback(
    Output('ctdl-dropdown-day', 'value'),
    Output('nmtd-dropdown-day', 'value'),
    Output('year-dropdown-day', 'value'),
    Output('month-dropdown-day', 'value'),
    Output('day-dropdown-day', 'value'),
    Input('tabs-main', 'value'),
    State('ctdl-dropdown-month', 'value'),
    State('nmtd-dropdown-month', 'value'),
    State('year-dropdown-month', 'value'),
    State('month-dropdown-month', 'value'),
)
def sync_to_day(tab, ctdl_m, nmtd_m, year_m, month_m):
    if tab == 'tab-day':
        return ctdl_m, nmtd_m, year_m, month_m, 1
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update



# --- Đồng bộ filter khi chuyển sang tab tháng ---
@app.callback(
    Output('ctdl-dropdown-month', 'value'),
    Output('nmtd-dropdown-month', 'value'),
    Output('year-dropdown-month', 'value'),
    Output('month-dropdown-month', 'value'),
    Input('tabs-main', 'value'),
    State('ctdl-dropdown-day', 'value'),
    State('nmtd-dropdown-day', 'value'),
    State('year-dropdown-day', 'value'),
    State('month-dropdown-day', 'value'),
)
def sync_to_month(tab, ctdl_d, nmtd_d, year_d, month_d):
    if tab == 'tab-month':
        return ctdl_d, nmtd_d, year_d, month_d
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update
# --- Callback cập nhật NMTD cho tab Tháng ---
@app.callback(
    Output('nmtd-dropdown-month', 'options'),
    Output('nmtd-dropdown-month', 'value'),
    Output('nmtd-dropdown-month', 'disabled'),
    Input('ctdl-dropdown-month', 'value')
)
def update_nmtd_month(selected_ctdl):
    if not selected_ctdl or df_sanluong.empty:
        return [], None, True

    try:
        # Kiểm tra CTDL có trong index không
        if selected_ctdl not in df_sanluong.index.get_level_values('CTDL'):
             print(f"CTDL '{selected_ctdl}' không tồn tại trong index (update_nmtd_month).")
             return [], None, True

        nmtd_options_list = sorted(df_sanluong.loc[selected_ctdl].index.get_level_values('NMTD').unique())
        nmtd_options = [{'label': n, 'value': n} for n in nmtd_options_list]

        if not nmtd_options:
            return [], None, True

        return nmtd_options, nmtd_options[0]['value'], False
    except Exception as e: # Bắt các lỗi khác, không chỉ KeyError
        print(f"Lỗi khi cập nhật NMTD tháng cho CTDL '{selected_ctdl}': {e}")
        return [], None, True

# --- Callback cập nhật NMTD cho tab Ngày ---
@app.callback(
    Output('nmtd-dropdown-day', 'options'),
    Output('nmtd-dropdown-day', 'value'),
    Output('nmtd-dropdown-day', 'disabled'),
    Input('ctdl-dropdown-day', 'value')
)
def update_nmtd_day(selected_ctdl):
    if not selected_ctdl or df_sanluong.empty:
        return [], None, True
    try:
        if selected_ctdl not in df_sanluong.index.get_level_values('CTDL'):
            print(f"CTDL '{selected_ctdl}' không tồn tại trong index (update_nmtd_day).")
            return [], None, True

        nmtd_options_list = sorted(df_sanluong.loc[selected_ctdl].index.get_level_values('NMTD').unique())
        nmtd_options = [{'label': n, 'value': n} for n in nmtd_options_list]
        if not nmtd_options:
            return [], None, True
        return nmtd_options, nmtd_options[0]['value'], False
    except Exception as e:
        print(f"Lỗi khi cập nhật NMTD ngày cho CTDL '{selected_ctdl}': {e}")
        return [], None, True

# --- Callback cập nhật Ngày cho tab Ngày ---
@app.callback(
    Output('day-dropdown-day', 'options'),
    Output('day-dropdown-day', 'value'),
    Output('day-dropdown-day', 'disabled'),
    Input('year-dropdown-day', 'value'),
    Input('month-dropdown-day', 'value'),
    State('day-dropdown-day', 'value') # Giữ giá trị ngày hiện tại để thử duy trì
)
def update_day_options_day(selected_year, selected_month, current_day_value):
    if not selected_year or not selected_month:
        return [], None, True # Vô hiệu hóa nếu chưa chọn Năm/Tháng

    try:
        max_day = calendar.monthrange(selected_year, selected_month)[1]
        day_options = [{'label': d, 'value': d} for d in range(1, max_day + 1)]

        new_day_value = None
        if current_day_value is not None and 1 <= current_day_value <= max_day:
            new_day_value = current_day_value
        elif day_options:
             new_day_value = day_options[0]['value'] # Ngày 1

        return day_options, new_day_value, False # Trả về options, giá trị mới, không vô hiệu hóa

    except Exception as e:
        print(f"Lỗi khi cập nhật ngày cho {selected_month}/{selected_year}: {e}")
        return [], None, True


# --- Callback chính: Cập nhật biểu đồ dựa trên tab và lựa chọn ---
@app.callback(
    Output('graph-output', 'figure'),
    Output('status-message', 'children'),
    Input('tabs-main', 'value'), # Input: Tab nào đang được chọn
    # Inputs từ Tab Tháng
    Input('ctdl-dropdown-month', 'value'),
    Input('nmtd-dropdown-month', 'value'),
    Input('year-dropdown-month', 'value'),
    Input('month-dropdown-month', 'value'),
    # Inputs từ Tab Ngày
    Input('ctdl-dropdown-day', 'value'),
    Input('nmtd-dropdown-day', 'value'),
    Input('year-dropdown-day', 'value'),
    Input('month-dropdown-day', 'value'),
    Input('day-dropdown-day', 'value')
)
def update_graph(active_tab,
                 ctdl_m, nmtd_m, year_m, month_m,
                 ctdl_d, nmtd_d, year_d, month_d, day_d):

    # Nếu DataFrame rỗng ngay từ đầu, không làm gì cả
    if df_sanluong.empty:
         return go.Figure(), "Lỗi: Không có dữ liệu đã xử lý hoặc xảy ra lỗi khi tải dữ liệu."

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial load'

    # Xác định giá trị dựa trên tab đang hoạt động HOẶC tab của input vừa thay đổi
    ctdl, nmtd, year, month, day, mode = None, None, None, None, None, None

    if active_tab == 'tab-month' or triggered_id in ['ctdl-dropdown-month', 'nmtd-dropdown-month', 'year-dropdown-month', 'month-dropdown-month']:
         # Ưu tiên xử lý cho tab tháng nếu input thuộc tab tháng hoặc tab tháng đang active
         if not all([ctdl_m, nmtd_m, year_m, month_m]):
             return dash.no_update, "Vui lòng chọn đủ CTDL, Nhà máy, Năm và Tháng."
         ctdl, nmtd, year, month, day = ctdl_m, nmtd_m, year_m, month_m, None
         mode = 'month'

    elif active_tab == 'tab-day' or triggered_id in ['ctdl-dropdown-day', 'nmtd-dropdown-day', 'year-dropdown-day', 'month-dropdown-day', 'day-dropdown-day']:
         # Ưu tiên xử lý cho tab ngày nếu input thuộc tab ngày hoặc tab ngày đang active
         if not all([ctdl_d, nmtd_d, year_d, month_d, day_d]):
              return dash.no_update, "Vui lòng chọn đủ CTDL, Nhà máy, Năm, Tháng và Ngày."
         ctdl, nmtd, year, month, day = ctdl_d, nmtd_d, year_d, month_d, day_d
         mode = 'day'

    # Nếu không xác định được mode (trường hợp hiếm), không cập nhật
    if mode is None:
        # Có thể vẽ biểu đồ mặc định ở đây nếu muốn, thay vì no_update
        # Ví dụ, vẽ cho tab tháng nếu có đủ thông tin
        if all([ctdl_m, nmtd_m, year_m, month_m]):
             ctdl, nmtd, year, month, day = ctdl_m, nmtd_m, year_m, month_m, None
             mode = 'month'
        else:
             return go.Figure(), "Vui lòng chọn các tùy chọn trên một tab."


    # --- Phần lọc và vẽ biểu đồ ---
    status = f"🔄 Đang lọc và vẽ biểu đồ cho {nmtd} - {mode}..."
    filtered_df = pd.DataFrame()
    title_ext = ""

    try:
        # Xây dựng tuple index để truy vấn
        if mode == 'month':
            idx_query = (ctdl, nmtd, year, month)
            title_ext = f"{month}/{year}"
        else: # mode == 'day'
            idx_query = (ctdl, nmtd, year, month, day)
            title_ext = f"{day}/{month}/{year}"

        # Kiểm tra sự tồn tại của index trước khi truy cập bằng .loc
        # Sử dụng pd.IndexSlice cho truy vấn rõ ràng hơn trên MultiIndex (tùy chọn, nhưng tốt)
        # slicer = pd.IndexSlice[idx_query] # Bỏ slicer nếu dùng tuple trực tiếp

        # Kiểm tra sự tồn tại của index con trước khi dùng loc
        # Cách 1: Kiểm tra trực tiếp tuple (đơn giản hơn)
        if idx_query in df_sanluong.index:
             # Lọc bằng .loc và chỉ lấy các cột cần thiết ngay lập tức
             # reset_index(drop=True) để bỏ MultiIndex cũ
             filtered_df = df_sanluong.loc[idx_query, ['TIME', 'CS', 'MADIEMDO']].reset_index(drop=True)
        # Cách 2: Dùng try-except (an toàn hơn nếu index phức tạp)
        # try:
        #      filtered_df = df_sanluong.loc[idx_query, ['TIME', 'CS', 'MADIEMDO']].reset_index(drop=True)
        # except KeyError:
        #      filtered_df = pd.DataFrame() # Không tìm thấy index

    except Exception as e: # Bắt tất cả lỗi, không chỉ KeyError
        status = f"❌ Lỗi khi lọc dữ liệu cho {idx_query}: {e}"
        print(status) # In lỗi ra console server để debug
        return go.Figure(), status # Trả về biểu đồ trống và thông báo

    if filtered_df.empty:
        status = f"ℹ️ Không có dữ liệu cho lựa chọn: {ctdl}, {nmtd}, {title_ext}."
        return go.Figure(), status

    # --- Vẽ biểu đồ ---
    try:
        filtered_df = filtered_df.sort_values('TIME') # Sắp xếp lại theo thời gian

        fig = px.line(
            filtered_df, x='TIME', y='CS', color='MADIEMDO',
            render_mode='webgl',
            markers=True if mode == 'day' else False
        )

        # Áp dụng layout y hệt như code gốc của bạn
        fig.update_layout(
            title=None,
            annotations=[
                dict(
                    text=f"<b>Công suất theo chu kỳ 30' - {nmtd} - {title_ext}</b>",
                    xref='paper', yref='paper', x=0.45, y=1.1,
                    xanchor='center', yanchor='top', showarrow=False,
                    font=dict(size=18, color='black')
                )
            ],
            template='plotly_white', height=600, hovermode='x unified',
            xaxis_title=None, yaxis_title='Giá trị công suất',
            xaxis=dict(
                showgrid=True, gridcolor='lightgrey', griddash='dot',
                title_font=dict(size=14, color='black'),
                ticklabelposition='outside', ticks='outside', ticklen=8
            ),
            yaxis=dict(
                showgrid=True, gridcolor='lightgrey', griddash='dot', rangemode='tozero',
                title_font=dict(size=14, color='black', weight='bold'),
                ticklabelposition='outside', ticks='outside', ticklen=8
            ),
            margin=dict(l=60, r=40, t=80, b=80),
            paper_bgcolor='white', plot_bgcolor='#f0f8ff'
        )
        fig.add_annotation(
            text="<b>Thời gian</b>", xref='paper', yref='paper', x=0.45, y=-0.15,
            showarrow=False, font=dict(size=14, color='black')
        )
        fig.update_traces(line=dict(width=2), hovertemplate='Thời gian: %{x}<br>Công suất: %{y}')

        status = f"✅ Hiển thị biểu đồ cho {nmtd} - {title_ext}"
        return fig, status

    except Exception as e:
        status = f"❌ Lỗi khi vẽ biểu đồ Plotly: {e}"
        print(status)
        return go.Figure(), status


# ==================== 5. CHẠY ỨNG DỤNG ====================
if __name__ == '__main__':
    # Kiểm tra lại nếu df_sanluong rỗng trước khi chạy server
    if df_sanluong.empty:
         print("\n❌ Không có dữ liệu hợp lệ để khởi chạy ứng dụng Dash.")
         # Bạn có thể thêm logic để hiển thị thông báo lỗi này trên giao diện web nếu muốn
         # Ví dụ: app.layout = html.Div("Lỗi tải dữ liệu, không thể khởi chạy ứng dụng.")
         # Hoặc để layout như cũ nhưng biểu đồ sẽ trống và có thông báo lỗi
         # Trong trường hợp này, ta vẫn chạy server để hiển thị lỗi (nếu có layout)
         print("\n🚀 Khởi chạy Dash server (với dữ liệu rỗng)...")
         print("Mở trình duyệt và truy cập: http://127.0.0.1:8050/")
         app.run(debug=True)
    else:
         print("\n🚀 Khởi chạy Dash server...")
         print("Mở trình duyệt và truy cập: http://127.0.0.1:8050/")
         app.run(debug=True) # Thay run_server thành run