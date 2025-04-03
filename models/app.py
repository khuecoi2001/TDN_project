
import pandas as pd
import os
import calendar
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

# ========== ĐỌC FILE PARQUET ==========
parquet_folder = r"C:\Khue\TDN\data\processed"
parquet_files = [os.path.join(parquet_folder, f)
                 for f in os.listdir(parquet_folder)
                 if f.startswith("sanluong_") and f.endswith(".parquet")]

cols = ['CTDL', 'NMTD', 'MADIEMDO', 'ENDTIME', 'CS']
df_all = pd.concat(
    (pd.read_parquet(f, columns=cols) for f in parquet_files),
    ignore_index=True
)

# ========== TIỀN XỬ LÝ ==========
df_sanluong = df_all.copy()
df_sanluong.rename(columns={'ENDTIME': 'TIME'}, inplace=True)
df_sanluong['YEAR'] = df_sanluong['TIME'].dt.year
df_sanluong['MONTH_NUM'] = df_sanluong['TIME'].dt.month
df_sanluong['DAY_NUM'] = df_sanluong['TIME'].dt.day

# ========== TẠO DROPDOWN ==========
def make_dropdown(options, description, width='250px', margin='0px 20px 0px 0px'):
    return widgets.Dropdown(
        options=options,
        description=description,
        layout=widgets.Layout(width=width, margin=margin),
        style={'description_width': 'auto'}
    )

ctdl_list = sorted(df_sanluong['CTDL'].dropna().unique())
year_list = sorted(df_sanluong['YEAR'].dropna().unique())

ctdl_dropdown_month = make_dropdown(ctdl_list, 'CTDL:', '300px')
nmtd_dropdown_month = make_dropdown([], 'Nhà máy:', '300px')
year_dropdown_month = make_dropdown(year_list, 'Năm:', '120px')
month_dropdown_month = make_dropdown(list(range(1, 13)), 'Tháng:', '120px')
day_placeholder = widgets.Label(value='', layout=widgets.Layout(width='120px', margin='0px 20px 0px 0px'))

ctdl_dropdown_day = make_dropdown(ctdl_list, 'CTDL:', '300px')
nmtd_dropdown_day = make_dropdown([], 'Nhà máy:', '300px')
year_dropdown_day = make_dropdown(year_list, 'Năm:', '120px')
month_dropdown_day = make_dropdown(list(range(1, 13)), 'Tháng:', '120px')
day_dropdown_day = make_dropdown(list(range(1, 32)), 'Ngày:', '120px')

month_widgets = [ctdl_dropdown_month, nmtd_dropdown_month, year_dropdown_month, month_dropdown_month]
day_widgets = [ctdl_dropdown_day, nmtd_dropdown_day, year_dropdown_day, month_dropdown_day, day_dropdown_day]

def update_nmtd(ctdl_value, dropdown):
    filtered = df_sanluong[df_sanluong['CTDL'] == ctdl_value]
    options = sorted(filtered['NMTD'].dropna().unique())
    dropdown.options = options

ctdl_dropdown_month.observe(lambda change: update_nmtd(change['new'], nmtd_dropdown_month), names='value')
ctdl_dropdown_day.observe(lambda change: update_nmtd(change['new'], nmtd_dropdown_day), names='value')

update_nmtd(ctdl_dropdown_month.value, nmtd_dropdown_month)
update_nmtd(ctdl_dropdown_day.value, nmtd_dropdown_day)

def update_day_options(*args):
    year = year_dropdown_day.value
    month = month_dropdown_day.value
    if year and month:
        max_day = calendar.monthrange(year, month)[1]
        day_dropdown_day.options = list(range(1, max_day + 1))

year_dropdown_day.observe(update_day_options, names='value')
month_dropdown_day.observe(update_day_options, names='value')
update_day_options()

# ========== VẼ BIỂU ĐỒ ==========
out = widgets.Output()

def plot_filtered(mode, ctdl, nmtd, year, month, day=None):
    if mode == 'month':
        filtered = df_sanluong[
            (df_sanluong['CTDL'] == ctdl) &
            (df_sanluong['NMTD'] == nmtd) &
            (df_sanluong['YEAR'] == year) &
            (df_sanluong['MONTH_NUM'] == month)
        ]
        title_ext = f"{month}/{year}"
    else:
        filtered = df_sanluong[
            (df_sanluong['CTDL'] == ctdl) &
            (df_sanluong['NMTD'] == nmtd) &
            (df_sanluong['YEAR'] == year) &
            (df_sanluong['MONTH_NUM'] == month) &
            (df_sanluong['DAY_NUM'] == day)
        ]
        title_ext = f"{day}/{month}/{year}"

    if filtered.empty:
        with out:
            out.clear_output()
            print("Không có dữ liệu cho lựa chọn này.")
        return

    filtered = filtered.sort_values('TIME')

    fig = px.line(
        filtered,
        x='TIME',
        y='CS',
        color='MADIEMDO',
        render_mode='webgl',
        markers=True if mode == 'day' else False
    )

    fig.update_layout(
        title=None,
        annotations=[
            dict(
                text=f"<b>Công suất theo chu kỳ 30' - {nmtd} - {title_ext}</b>",
                xref='paper', yref='paper',
                x=0.45, y=1.1,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=18, color='black')
            )
        ],
        template='plotly_white',
        height=600,
        hovermode='x unified',
        xaxis_title=None,
        yaxis_title='Giá trị công suất',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            griddash='dot',
            title_font=dict(size=14, color='black', weight='bold'),
            ticklabelposition='outside',
            ticks='outside',
            ticklen=8
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            griddash='dot',
            rangemode='tozero',
            title_font=dict(size=14, color='black', weight='bold'),
            ticklabelposition='outside',
            ticks='outside',
            ticklen=8
        ),
        margin=dict(l=60, r=40, t=80, b=80),
        paper_bgcolor='white',
        plot_bgcolor='#f0f8ff'
    )

    fig.add_annotation(
        text="<b>Thời gian</b>",
        xref='paper', yref='paper',
        x=0.45, y=-0.15,
        showarrow=False,
        font=dict(size=14, color='black')
    )

    fig.update_traces(
        line=dict(width=2),
        hovertemplate='Thời gian: %{x}<br>Công suất: %{y}'
    )

    with out:
        out.clear_output()
        fig.show()

# ========== EVENT ==========
def on_change_month(change):
    plot_filtered('month', ctdl_dropdown_month.value, nmtd_dropdown_month.value,
                  year_dropdown_month.value, month_dropdown_month.value)

def on_change_day(change):
    plot_filtered('day', ctdl_dropdown_day.value, nmtd_dropdown_day.value,
                  year_dropdown_day.value, month_dropdown_day.value, day_dropdown_day.value)

def disable_observe(widgets, callback):
    for w in widgets:
        w.unobserve(callback, names='value')

def enable_observe(widgets, callback):
    for w in widgets:
        w.observe(callback, names='value')

for w in month_widgets:
    w.observe(on_change_month, 'value')
for w in day_widgets:
    w.observe(on_change_day, 'value')

tab_thang = widgets.HBox([ctdl_dropdown_month, nmtd_dropdown_month,
                         year_dropdown_month, month_dropdown_month, day_placeholder],
                         layout=widgets.Layout(justify_content='flex-start', overflow='visible'))

tab_ngay = widgets.HBox([ctdl_dropdown_day, nmtd_dropdown_day,
                        year_dropdown_day, month_dropdown_day, day_dropdown_day],
                        layout=widgets.Layout(justify_content='flex-start', overflow='visible'))

tabs = widgets.Tab(children=[tab_thang, tab_ngay])
tabs.set_title(0, 'Xem theo tháng')
tabs.set_title(1, 'Xem theo ngày')

def on_tab_change(change):
    if change['name'] == 'selected_index':
        if change['new'] == 1:
            disable_observe(day_widgets, on_change_day)
            ctdl_dropdown_day.value = ctdl_dropdown_month.value
            nmtd_dropdown_day.value = nmtd_dropdown_month.value
            year_dropdown_day.value = year_dropdown_month.value
            month_dropdown_day.value = month_dropdown_month.value
            day_dropdown_day.value = 1
            enable_observe(day_widgets, on_change_day)
            on_change_day(None)
        elif change['new'] == 0:
            disable_observe(month_widgets, on_change_month)
            ctdl_dropdown_month.value = ctdl_dropdown_day.value
            nmtd_dropdown_month.value = nmtd_dropdown_day.value
            year_dropdown_month.value = year_dropdown_day.value
            month_dropdown_month.value = month_dropdown_day.value
            enable_observe(month_widgets, on_change_month)
            on_change_month(None)

tabs.observe(on_tab_change)

# ========== HIỂN THỊ ==========
display(tabs, out)
