import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

def set_dark_theme():
    custom_rc = {
        "axes.facecolor": "#050814",     
        "figure.facecolor": "#050814",   
        "figure.edgecolor": "#050814",
        "grid.color": "#1E293B",  
        "grid.linestyle": "--",
        "text.color": "#F1F5F9",        
        "axes.labelcolor": "#F1F5F9",    
        "xtick.color": "#F1F5F9",        
        "ytick.color": "#F1F5F9",        
        "axes.edgecolor": "#1E293B",     
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.sans-serif": ["IBM Plex Sans", "Arial", "sans-serif"]
    }
    sns.set_theme(style="darkgrid", rc=custom_rc)

@st.cache_data
def load_data():
    X_train = pd.read_csv("data/data_for_scripts/X_train.csv")
    try:
        Y_train = pd.read_csv("data/data_for_scripts/Y_train.csv")
    except FileNotFoundError:
        Y_train = pd.DataFrame({"id": X_train["id"]})
        for i in range(1, 7):
            Y_train[f"attr_{i}"] = np.nan
    return X_train, Y_train

@st.cache_data
def process_sequences(_df):
    sequences = []
    for _, row in _df.drop("id", axis=1, errors="ignore").iterrows():
        seq = [int(x) for x in row.values if not pd.isna(x)]
        if seq:
            sequences.append(seq)
            
    first_events = [seq[0] for seq in sequences]
    last_events = [seq[-1] for seq in sequences]
    
    bigrams = []
    trigrams = []
    for s in sequences:
        for i in range(len(s) - 1):
            bigrams.append(tuple(s[i:i+2]))
        for i in range(len(s) - 2):
            trigrams.append(tuple(s[i:i+3]))
            
    return sequences, first_events, last_events, bigrams, trigrams

@st.cache_data
def precompute_counts(events):
    return pd.Series(events).value_counts()

def get_pie_data_slice(counts_series, k):
    top_k_series = counts_series.head(k).copy()
    other_sum = counts_series.iloc[k:].sum()
    if other_sum > 0:
        top_k_series["Other"] = other_sum
    df = top_k_series.reset_index()
    df.columns = ["Mã hành động", "Số lượng"]
    df["Mã hành động"] = df["Mã hành động"].astype(str) 
    return df

@st.cache_data
def precompute_ytrain_stats(_Y_train):
    attrs = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
    desc_stats = _Y_train[attrs].agg(["min", "max", "mean", "nunique"]).T
    return desc_stats

def generate_histogram_fig(_seq_lengths):
    set_dark_theme()
    fig_hist, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(_seq_lengths, bins=range(1, 68), kde=True, color="#3B82F6", ax=ax1, edgecolor="#050814")
    ax1.set_xlabel("Số lượng hành động trong một phiên", fontsize=14, labelpad=10)
    ax1.set_ylabel("Số lượng khách hàng", fontsize=14, labelpad=10)
    plt.tight_layout()
    return fig_hist

def generate_attr_distribution_fig(Y_train):
    set_dark_theme()
    attrs = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
    fig_dist, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, attr in enumerate(attrs):
        top_vals = Y_train[attr].value_counts().head(20).reset_index()
        top_vals.columns = [attr, "count"]
        
        sns.barplot(data=top_vals, x=attr, y="count", ax=axes[i], 
                    hue=attr, palette="mako", legend=False)
        
        axes[i].set_title(f"Phân phối Top 20 của {attr}", fontsize=14, pad=10, color="#F1F5F9")
        axes[i].set_xlabel("Giá trị mã")
        axes[i].set_ylabel("Số lượng")
        
    plt.tight_layout()
    return fig_dist

def generate_transition_matrix_fig(sequences):
    set_dark_theme()
    all_actions = [a for s in sequences for a in s]
    counts = Counter(all_actions)
    top_15 = [a for a, _ in counts.most_common(15)]
    
    transitions = []
    for s in sequences:
        for i in range(len(s) - 1):
            transitions.append((s[i], s[i+1]))
            
    trans_df = pd.DataFrame(transitions, columns=["From", "To"])
    matrix = pd.crosstab(trans_df["From"], trans_df["To"], normalize="index")
    matrix_filtered = matrix.loc[matrix.index.isin(top_15), matrix.columns.isin(top_15)]
    
    fig_matrix, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix_filtered, annot=True, fmt=".2f", cmap="mako", ax=ax, 
                cbar_kws={"label": "Xác suất chuyển đổi"})
    
    ax.set_xlabel("Hành động tiếp theo", fontsize=12, labelpad=10)
    ax.set_ylabel("Hành động hiện tại", fontsize=12, labelpad=10)
    ax.tick_params(colors="#F1F5F9") 
    plt.tight_layout()
    return fig_matrix

def generate_seasonality_fig(Y_train):
    set_dark_theme()
    daily_capacity = Y_train.groupby("attr_2")[["attr_3", "attr_6"]].mean().reset_index()
    
    fig_season, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(x="attr_2", y="attr_3", data=daily_capacity, marker="o", color="#F43F5E", linewidth=2.5, label="Khâu vận hành A (attr_3)", ax=ax)
    sns.lineplot(x="attr_2", y="attr_6", data=daily_capacity, marker="s", color="#3B82F6", linewidth=2.5, label="Khâu vận hành B (attr_6)", ax=ax)
    
    ax.set_title("Biến động công suất theo ngày trong tháng", fontsize=16, pad=15, color="#F1F5F9")
    ax.set_xlabel("Ngày bắt đầu giao dịch (attr_2: từ mùng 1 đến 31)", fontsize=14)
    ax.set_ylabel("Công suất trung bình", fontsize=14)
    ax.set_xticks(np.arange(1, 32, step=1))
    
    legend = ax.legend(fontsize=12, facecolor="#111827", edgecolor="#1E293B")
    for text in legend.get_texts():
        text.set_color("#F1F5F9")
        
    ax.axvspan(25, 31, color="#F59E0B", alpha=0.15, label="Vùng Dồn đơn")
    
    plt.tight_layout()
    return fig_season

def generate_lift_score_fig(X_train, Y_train):
    set_dark_theme()
    Y_temp = Y_train[["id", "attr_3", "attr_6"]].copy()
    p90_attr3 = Y_temp["attr_3"].quantile(0.90)
    p90_attr6 = Y_temp["attr_6"].quantile(0.90)
    
    Y_temp["is_extreme_attr3"] = Y_temp["attr_3"] >= p90_attr3
    Y_temp["is_extreme_attr6"] = Y_temp["attr_6"] >= p90_attr6
    
    df_merged = pd.merge(X_train, Y_temp[["id", "is_extreme_attr3", "is_extreme_attr6"]], on="id")
    feature_cols = [col for col in X_train.columns if col.startswith("feature_")]
    
    all_acts = df_merged[feature_cols].values.flatten()
    action_counts = pd.Series(all_acts[~np.isnan(all_acts)]).value_counts()
    valid_actions = action_counts[action_counts >= 100].index
    
    def calculate_lift_scores(df, target_col):
        def get_probs(subset):
            actions = subset[feature_cols].values.flatten()
            valid_actions_in_subset = actions[~np.isnan(actions) & (actions != 0)]
            return pd.Series(valid_actions_in_subset).value_counts(normalize=True)
        
        prob_normal = get_probs(df[~df[target_col]])
        prob_extreme = get_probs(df[df[target_col]])
        
        lift_df = pd.DataFrame({"P_Extreme": prob_extreme, "P_Normal": prob_normal}).fillna(0)
        lift_df = lift_df.loc[lift_df.index.isin(valid_actions)]
        
        epsilon = 1e-8
        lift_df["Lift_Score"] = lift_df["P_Extreme"] / (lift_df["P_Normal"] + epsilon)
        return lift_df.sort_values("Lift_Score", ascending=False)
    
    lift_attr3 = calculate_lift_scores(df_merged, "is_extreme_attr3")
    lift_attr6 = calculate_lift_scores(df_merged, "is_extreme_attr6")
    
    top_attr3 = lift_attr3.head(11)
    top_attr6 = lift_attr6.head(11)
    
    fig_lift, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.barplot(x=top_attr3["Lift_Score"], y=top_attr3.index.astype(int).astype(str), palette="flare", ax=axes[0])
    axes[0].set_title("Khâu vận hành A (attr_3)", fontsize=16, pad=15)
    axes[0].set_xlabel("Hệ số nâng", fontsize=12)
    axes[0].set_ylabel("Mã hành động", fontsize=12)
    axes[0].axvline(x=1.0, color="#64748B", linestyle="--", linewidth=2)
    for p in axes[0].patches:
        # Màu annotation đổi sang màu hồng nhạt để nổi trên nền đen
        axes[0].annotate(f"{p.get_width():.2f}x", (p.get_width() + 0.1, p.get_y() + p.get_height() / 2.), 
                         ha="left", va="center", fontsize=11, color="#FDA4AF", fontweight="bold")
                         
    sns.barplot(x=top_attr6["Lift_Score"], y=top_attr6.index.astype(int).astype(str), palette="crest", ax=axes[1])
    axes[1].set_title("Khâu vận hành B (attr_6)", fontsize=16, pad=15)
    axes[1].set_xlabel("Hệ số nâng", fontsize=12)
    axes[1].set_ylabel("Mã hành động", fontsize=12)
    axes[1].axvline(x=1.0, color="#64748B", linestyle="--", linewidth=2)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_width():.2f}x", (p.get_width() + 0.1, p.get_y() + p.get_height() / 2.), 
                         ha="left", va="center", fontsize=11, color="#93C5FD", fontweight="bold")
                         
    plt.tight_layout()
    return fig_lift

@st.cache_resource
def generate_interactive_pie(_counts, title, colorscale):
    fig = go.Figure()
    for k in range(1, 13):
        df = get_pie_data_slice(_counts, k)
        fig.add_trace(
            go.Pie(
                labels=df["Mã hành động"], 
                values=df["Số lượng"], 
                name=f"Top {k}",
                visible=(k==5),
                textinfo="percent+label",
                marker=dict(colors=px.colors.qualitative.Set3 if colorscale == "Set3" else px.colors.qualitative.Pastel)
            )
        )
    steps = []
    for i in range(12):
        visibility = [False] * 12
        visibility[i] = True
        step = dict(
            method="restyle",
            args=[{"visible": visibility}],
            label=str(i + 1)
        )
        steps.append(step)
    sliders = [dict(
        active=4, 
        currentvalue={"prefix": "Top K hành động: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders, 
        title=f"{title} (Top K)", 
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="#F1F5F9")    
    )
    return fig

@st.cache_data
def preload_data():
    X_train, Y_train = load_data()
    sequences, first_events, last_events, bigrams, trigrams = process_sequences(X_train)
    first_counts = precompute_counts(first_events)
    last_counts = precompute_counts(last_events)

    top_bi = Counter(bigrams).most_common(10)
    bi_df = pd.DataFrame([{"Chuỗi hành động": str(p), "Số lần xuất hiện": c} for p, c in top_bi])
    top_tri = Counter(trigrams).most_common(10)
    tri_df = pd.DataFrame([{"Chuỗi hành động": str(p), "Số lần xuất hiện": c} for p, c in top_tri])

    features = [col for col in X_train.columns if col.startswith("feature_")]
    seq_lengths = X_train[features].notna().sum(axis=1)

    desc_stats = precompute_ytrain_stats(Y_train)

    cached_histogram = generate_histogram_fig(seq_lengths)
    cached_attr_dist_fig = generate_attr_distribution_fig(Y_train)
    cached_transition_matrix = generate_transition_matrix_fig(sequences)
    
    cached_seasonality_fig = generate_seasonality_fig(Y_train)
    cached_lift_score_fig = generate_lift_score_fig(X_train, Y_train)
    
    cached_interactive_pie_first = generate_interactive_pie(first_counts, "Sự kiện mở đầu", "Set3")
    cached_interactive_pie_last = generate_interactive_pie(last_counts, "Sự kiện kết thúc", "Pastel")

    merged_data = pd.merge(X_train, Y_train, on="id", how="left")
    
    return (cached_histogram, cached_attr_dist_fig, cached_transition_matrix, 
            cached_seasonality_fig, cached_lift_score_fig, 
            cached_interactive_pie_first, cached_interactive_pie_last, 
            bi_df, tri_df, desc_stats, merged_data)

import streamlit as st
import pandas as pd
import numpy as np

def show():
    (cached_histogram, cached_attr_dist_fig, cached_transition_matrix, 
     cached_seasonality_fig, cached_lift_score_fig, 
     cached_interactive_pie_first, cached_interactive_pie_last, 
     bi_df, tri_df, desc_stats, merged_data) = preload_data()

    st.markdown("""
        <style>
        /* Global Page Feel & Font System */
        [data-testid="stAppViewContainer"] {
            background-color: #030610;
            background-image: 
                radial-gradient(at 50% 0%, rgba(30, 58, 138, 0.25) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(3, 6, 16, 0.5) !important;
            backdrop-filter: blur(12px) !important;
        }
        [data-testid="stMainBlockContainer"] {
            padding-top: 3rem;
        }
        html, body, [class*="css"]  {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(59, 130, 246, 0.8);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }
                
        /* Các khối có viền, nền và đổ bóng chuẩn SaaS */
        .saas-card {
            background-color: #0c1223;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .saas-card:hover {
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
            border-color: #3B82F6;
        }

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 0px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: transparent;
            border-radius: 0px;
            padding: 10px 4px;
            font-weight: 600;
            font-size: 0.95rem;
            color: #94A3B8;
            transition: color 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            color: #F8FAFC !important;
            border-bottom: 2px solid #3B82F6 !important;
        }
        .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
            color: #CBD5E1;
        }

        /* Metrics Cards Styling */
        div[data-testid="metric-container"] {
            background-color: #0c1223; /* Đã phục hồi nền tối */
            border: 1px solid #1E293B; /* Đã phục hồi viền bao */
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #3B82F6; /* Viền nhấn màu xanh */
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        }
        div[data-testid="stMetricLabel"] > div {
            color: #94A3B8 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricValue"] > div {
            color: #F8FAFC !important;
            font-weight: 700 !important;
            font-size: 1.8rem !important;
        }

        /* Typography */
        .section-title {
            color: #F8FAFC;
            font-weight: 700;
            font-size: 2rem;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
            text-align: left;
        }
        .section-subtitle {
            color: #64748B;
            font-size: 1rem;
            margin-bottom: 2rem;
            display: block;
            text-align: left;
        }
        .chart-title {
            color: #E2E8F0;
            font-weight: 600;
            font-size: 1.1rem;
            margin-top: 0px;
            margin-bottom: 8px;
        }
        .chart-desc {
            color: #64748B;
            font-size: 0.9rem;
            display: block;
            margin-bottom: 1rem;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)

    col_left, col_main, col_right = st.columns([0.05, 0.9, 0.05])

    with col_main:
        # --- HEADER ---
        st.markdown("<h1 class='section-title'>Trung tâm Giám sát Vận hành</h1>", unsafe_allow_html=True)
        st.markdown("<span class='section-subtitle'>Nền tảng phân tích hành vi người dùng và đánh giá hiệu suất hệ thống thời gian thực.</span>", unsafe_allow_html=True)

        # --- KPI METRICS ---
        total_sessions = len(merged_data)
        features = [col for col in merged_data.columns if col.startswith("feature_")]
        avg_seq_len = merged_data[features].notna().sum(axis=1).mean()
        
        # KHÔNG DÙNG st.container() bọc ngoài nữa. Streamlit sẽ tự tạo box nhờ CSS Metric.
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Tổng phiên truy cập", value=f"{total_sessions:,}")
        with col2:
            st.metric(label="TB thao tác / phiên", value=f"{avg_seq_len:.2f}")
        with col3:
            st.metric(label="Tải trọng kho bãi (A)", value="Cảnh báo", delta="Dồn đơn cuối tháng", delta_color="inverse")
        with col4:
            st.metric(label="Hệ thống giao vận (B)", value="Ổn định", delta="Trong ngưỡng", delta_color="normal")

        st.write("")
        st.write("")

        # --- TABS LAYOUT ---
        tab_overview, tab_behavior, tab_data = st.tabs([
            "Tổng quan hệ thống", 
            "Phân tích hành vi", 
            "Truy vấn dữ liệu"
        ])

        # ==========================================
        # TAB 1: TỔNG QUAN
        # ==========================================
        with tab_overview:
            st.write("")
            col_chart1, col_chart2 = st.columns([1, 1])
            with col_chart1:
                # Đổi st.container(border=True) thành div HTML bọc lại
                st.markdown("""
                <div class='saas-card'>
                    <p class='chart-title'>Phân bố độ dài chuỗi hành động</p>
                    <span class='chart-desc'>Đánh giá mức độ tương tác sâu của người dùng trên nền tảng trực tuyến.</span>
                """, unsafe_allow_html=True)
                st.pyplot(cached_histogram, transparent=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_chart2:
                st.markdown("""
                <div class='saas-card'>
                    <p class='chart-title'>Chu kỳ áp lực vận hành</p>
                    <span class='chart-desc'>Biến động công suất theo thời gian thực tế trong tháng.</span>
                """, unsafe_allow_html=True)
                st.pyplot(cached_seasonality_fig, transparent=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='saas-card'>
                <p class='chart-title'>Phân phối đặc trưng hệ thống</p>
                <span class='chart-desc'>Thống kê Top 20 giá trị phổ biến nhất của các thuộc tính vận hành.</span>
            """, unsafe_allow_html=True)
            st.pyplot(cached_attr_dist_fig, transparent=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("Bảng thống kê mô tả chi tiết", expanded=False):
                st.dataframe(desc_stats, use_container_width=True)

        # ==========================================
        # TAB 2: HÀNH VI KHÁCH HÀNG
        # ==========================================
        with tab_behavior:
            st.write("")
            
            st.markdown("""
            <div class='saas-card'>
                <p class='chart-title'>Phân bổ điểm chạm đầu & cuối</p>
                <span class='chart-desc'>Nhận diện luồng truy cập (Entry point) và điểm thoát (Exit point) phổ biến nhất của người dùng.</span>
            """, unsafe_allow_html=True)
            col_pie1, col_pie2 = st.columns(2)
            with col_pie1:
                st.plotly_chart(cached_interactive_pie_first, use_container_width=True)
            with col_pie2:
                st.plotly_chart(cached_interactive_pie_last, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='saas-card'>
                <p class='chart-title'>Đánh giá rủi ro hành vi</p>
                <span class='chart-desc'>Đo lường hệ số nâng (Lift Score) của các mã hành động làm tăng đột biến nguy cơ quá tải hệ thống (>90% công suất).</span>
            """, unsafe_allow_html=True)
            st.pyplot(cached_lift_score_fig, transparent=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            col_matrix, col_seq = st.columns([1.2, 1])
            with col_matrix:
                st.markdown("""
                <div class='saas-card'>
                    <p class='chart-title'>Ma trận chuyển đổi trạng thái</p>
                    <span class='chart-desc'>Xác suất dịch chuyển giữa các bước thao tác (Markov Chain logic).</span>
                """, unsafe_allow_html=True)
                st.pyplot(cached_transition_matrix, transparent=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_seq:
                st.markdown("""
                <div class='saas-card'>
                    <p class='chart-title'>Khai phá chuỗi tuần tự phổ biến</p>
                    <span class='chart-desc'>Tần suất xuất hiện của các cụm thao tác (N-grams) điển hình.</span>
                """, unsafe_allow_html=True)
                
                max_bi = int(bi_df["Số lần xuất hiện"].max()) if not bi_df.empty else 100
                max_tri = int(tri_df["Số lần xuất hiện"].max()) if not tri_df.empty else 100

                st.markdown("**Top 10 Bigrams (Chuỗi 2 bước)**")
                st.dataframe(
                    bi_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Số lần xuất hiện": st.column_config.ProgressColumn(
                            "Tần suất",
                            help="Số lượng phiên chứa chuỗi hành động này",
                            format="%f",
                            min_value=0,
                            max_value=max_bi,
                        ),
                    }
                )
                
                st.markdown("**Top 10 Trigrams (Chuỗi 3 bước)**")
                st.dataframe(
                    tri_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Số lần xuất hiện": st.column_config.ProgressColumn(
                            "Tần suất",
                            help="Số lượng phiên chứa chuỗi hành động này",
                            format="%f",
                            min_value=0,
                            max_value=max_tri,
                        ),
                    }
                )
                st.markdown("</div>", unsafe_allow_html=True)

        # ==========================================
        # TAB 3: DỮ LIỆU THÔ
        # ==========================================
        with tab_data:
            st.write("")
            st.markdown("""
            <div class='saas-card'>
                <p class='chart-title'>Trích xuất dữ liệu hệ thống</p>
                <span class='chart-desc'>Dữ liệu nhật ký đã được gộp và tiền xử lý (hiển thị 70 bản ghi đầu tiên).</span>
            """, unsafe_allow_html=True)
            st.dataframe(merged_data.head(70), height=500, use_container_width=True, hide_index=True)
            
            st.write("")
            col_btn, _ = st.columns([1, 4])
            with col_btn:
                csv = merged_data.head(500).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải xuống Sample Data (CSV)",
                    data=csv,
                    file_name='ecommerce_log_data.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)