import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import calendar

def parse_predictions(df_preds):
    orders = []
    base_year = 2026
    
    for _, row in df_preds.iterrows():
        order_id = row.get("id", f"ORD_{np.random.randint(1000, 9999)}")
        
        def safe_date(m_float, d_float):
            m = int(np.clip(round(m_float), 1, 12))
            max_days = calendar.monthrange(base_year, m)[1]
            d = int(np.clip(round(d_float), 1, max_days))
            return datetime(base_year, m, d).date()

        start_date = safe_date(row["attr_1"], row["attr_2"])
        pred_end_date = safe_date(row["attr_4"], row["attr_5"])
        
        if pred_end_date < start_date:
            pred_end_date = start_date + timedelta(days=1)
            
        total_work = max(5.0, abs(row["attr_6"] - row["attr_3"]))
        
        orders.append({
            "Order_ID": order_id,
            "Start_Date": start_date, 
            "Pred_End_Date": pred_end_date,
            "Total_Work": total_work
        })
        
    return pd.DataFrame(orders)

def optimize_schedule(df_orders, k_factories):
    FACTORY_CAPACITY = 100.0 
    
    schedule_records = []
    delayed_orders = []
    daily_usage = {i: {} for i in range(1, k_factories + 1)}
    
    df_orders = df_orders.sort_values(by="Pred_End_Date")
    
    for _, row in df_orders.iterrows():
        order_id = row["Order_ID"]
        start_date = row["Start_Date"]
        pred_end_date = row["Pred_End_Date"]
        total_work_needed = row["Total_Work"]
        
        best_factory_id = 1
        earliest_possible_finish = datetime(2099, 12, 31).date() 
        
        for fac_id in range(1, k_factories + 1):
            scan_date = start_date
            temp_work = total_work_needed
            
            while temp_work > 0:
                if scan_date not in daily_usage[fac_id]:
                    daily_usage[fac_id][scan_date] = 0.0
                
                avail_cap = FACTORY_CAPACITY - daily_usage[fac_id][scan_date]
                if avail_cap > 0:
                    temp_work -= avail_cap
                
                if temp_work > 0:
                    scan_date += timedelta(days=1)
            
            if scan_date < earliest_possible_finish:
                earliest_possible_finish = scan_date
                best_factory_id = fac_id

        current_scan_date = start_date
        work_left = total_work_needed
        
        while work_left > 0:
            if current_scan_date not in daily_usage[best_factory_id]:
                daily_usage[best_factory_id][current_scan_date] = 0.0
                
            avail_cap = FACTORY_CAPACITY - daily_usage[best_factory_id][current_scan_date]
            
            if avail_cap > 0:
                work_to_do = min(work_left, avail_cap)
                schedule_records.append({
                    "Date": current_scan_date,
                    "Order_ID": order_id,
                    "Factory_ID": best_factory_id, 
                    "Work_Percent": work_to_do
                })
                daily_usage[best_factory_id][current_scan_date] += work_to_do
                work_left -= work_to_do
            
            if work_left > 0:
                current_scan_date += timedelta(days=1)
                
        actual_end_date = current_scan_date
        
        if actual_end_date > pred_end_date:
            delay_days = (actual_end_date - pred_end_date).days
            delayed_orders.append({
                "Order_ID": order_id,
                "Tổng Công suất": round(row["Total_Work"], 1),
                "Nhà máy": f"Factory {best_factory_id}",
                "Ngày dự kiến hoàn thành": pred_end_date.strftime("%Y-%m-%d"),
                "Ngày thực tế hoàn thành": actual_end_date.strftime("%Y-%m-%d"),
                "Trễ hẹn (Ngày)": delay_days
            })

    return pd.DataFrame(schedule_records), pd.DataFrame(delayed_orders)

@st.cache_data 
def format_export_data(_df_schedule):
    export_list = []
    
    grouped = _df_schedule.groupby("Order_ID")
    for order_id, group in grouped:
        row_dict = {"Order_ID": order_id}
        group = group.sort_values(by="Date")
        
        factory_id = group.iloc[0]["Factory_ID"]
        row_dict["Assigned_Factory"] = f"Nhà máy {factory_id}"
        
        for idx, (_, r) in enumerate(group.iterrows()):
            col_name = f"Plan_{idx+1}" 
            date_str = r["Date"].strftime("%Y-%m-%d")
            work_val = round(r["Work_Percent"], 2)
            
            row_dict[col_name] = f"{date_str} ({work_val}%)"
            
        export_list.append(row_dict)
        
    df_export = pd.DataFrame(export_list)
    df_export = df_export.fillna("")
    
    cols = df_export.columns.tolist()
    cols.insert(1, cols.pop(cols.index("Assigned_Factory")))
    df_export = df_export[cols]
    
    return df_export

def show():
    st.markdown("""
        <style>
        /* Overall Page Feel */
        /* 1. Overall Page Feel: Radial Glow (Modern AI SaaS) */
        [data-testid="stAppViewContainer"] {
            background-color: #030610;
            background-image: 
                radial-gradient(at 50% 0%, rgba(30, 58, 138, 0.25) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: #e2e8f0;
        }
        [data-testid="stHeader"] {
            background-color: rgba(3, 6, 16, 0.5) !important;
            backdrop-filter: blur(12px) !important; /* Làm mờ thanh header trên cùng */
        }
        [data-testid="stMainBlockContainer"] {
            padding-top: 3rem;
        }

        /* Typography */
        .section-title {
            color: #F8FAFC;
            font-weight: 700;
            margin-bottom: 0.2rem;
            text-align: left;
        }
        .section-subtitle {
            color: #64748B;
            font-size: 1rem;
            margin-bottom: 2rem;
            text-align: left;
            display: block;
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
        }
        
        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #0c1223;
            border: 1px solid #1E293B;
            padding: 1.2rem;
            border-radius: 8px;
            border-left: 4px solid #3B82F6; 
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
        
        /* Radio button styling */
        .stRadio [role="radiogroup"] {
            gap: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    st.markdown("<h1 class='section-title'>Xây dựng lịch trình vận hành tối ưu</h1>", unsafe_allow_html=True)
    st.markdown("<span class='section-subtitle'>Hệ thống tự động phân bổ khối lượng công việc vào nhiều nhà máy, theo dõi tải trọng riêng biệt và xuất file kế hoạch chi tiết.</span>", unsafe_allow_html=True)

    # ==========================================
    # PHẦN 1: CẤU HÌNH HỆ THỐNG (INPUT)
    # ==========================================
    with st.container(border=True):
        st.markdown("<p class='chart-title'>Thiết lập dữ liệu & năng lực sản xuất</p>", unsafe_allow_html=True)
        st.write("")
        
        col_data, col_config = st.columns([1.5, 1])
        
        df_raw_preds = None
        
        with col_data:
            st.markdown("**1. Nguồn dữ liệu đơn hàng**")
            data_source = st.radio(
                "Chọn nguồn dữ liệu đầu vào:",
                ["Lấy kết quả vừa chạy ở phần Dự đoán hàng loạt", "Tải lên tệp thông tin đơn hàng mới"],
                horizontal=False,
                label_visibility="collapsed"
            )
            
            if "Lấy kết quả vừa chạy" in data_source:
                if "batch_preds" in st.session_state:
                    df_raw_preds = st.session_state.batch_preds
                    st.success(f"Tải thành công {len(df_raw_preds)} đơn hàng từ bộ nhớ tạm.")
                else:
                    st.warning("Chưa có dữ liệu trong bộ nhớ. Vui lòng chạy mô hình trước hoặc chọn tải file mới lên.")
            else:
                uploaded_file = st.file_uploader("Tải lên tệp thông tin đơn hàng (chứa 6 thuộc tính)", type=["csv"])
                if uploaded_file is not None:
                    df_raw_preds = pd.read_csv(uploaded_file)
                    st.success(f"Tải thành công file chứa {len(df_raw_preds)} đơn hàng.")

        with col_config:
            st.markdown("**2. Thông số Nhà máy**")
            k_factories_input = st.number_input(
                "Nhập số lượng nhà máy (K) hiện có:", 
                min_value=1, max_value=10000, value=10, step=1,
                help="Mặc định: 100 công suất/nhà máy/ngày"
            )
            
            st.write("")
            with st.form("optimize_form", border=False):
                    submit_btn = st.form_submit_button("Xây dựng lịch trình tối ưu", type="primary", use_container_width=True)
                    
                    if submit_btn:
                        if df_raw_preds is None:
                            st.error("Chưa có dữ liệu đầu vào!")
                        else:
                            # --- BỔ SUNG: BƯỚC KIỂM TRA DỮ LIỆU HỢP LỆ ---
                            required_cols = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6']
                            missing_cols = [col for col in required_cols if col not in df_raw_preds.columns]
                            
                            if missing_cols:
                                # Báo lỗi UI chuyên nghiệp thay vì sập app
                                st.error(f"Dữ liệu không hợp lệ! Tệp của bạn đang thiếu các cột kết quả dự báo: {', '.join(missing_cols)}")
                                st.info("Gợi ý: Hãy chắc chắn bạn đã chọn đúng tệp Kết quả (result.csv) được tải về từ trang 'Dự đoán và Giải thích'.")
                            else:
                                # Nếu dữ liệu chuẩn, tiến hành chạy thuật toán
                                with st.spinner("Đang tính toán bài toán tối ưu..."):
                                    start_time = time.time()
                                    
                                    df_orders = parse_predictions(df_raw_preds)
                                    df_schedule, df_delayed = optimize_schedule(df_orders, k_factories_input) 
                                    
                                    st.session_state.df_schedule = df_schedule
                                    st.session_state.df_delayed = df_delayed
                                    st.session_state.total_orders = len(df_orders)
                                    st.session_state.run_k_factories = k_factories_input
                                    st.session_state.inference_time = time.time() - start_time
                                    
                                    st.rerun()
    # ==========================================
    # PHẦN 2: KẾT QUẢ ĐẦU RA (OUTPUT)
    # ==========================================
    if "df_schedule" in st.session_state:
        df_schedule = st.session_state.df_schedule
        df_delayed = st.session_state.df_delayed
        total_orders = st.session_state.total_orders
        inf_time = st.session_state.inference_time
        k_facs = st.session_state.get("run_k_factories", k_factories_input) 
        
        st.write("")
        
        # --- BÁO CÁO TỔNG QUAN ---
        with st.container(border=True):
            col_title, col_time = st.columns([3, 1])
            with col_title:
                st.markdown("<p class='chart-title'>Báo cáo tổng quan tối ưu hóa</p>", unsafe_allow_html=True)
            with col_time:
                st.markdown(f"<p style='text-align: right; color: #64748B; font-size: 0.9em; margin-top: 5px;'>⏱️ Thời gian xử lý: {inf_time:.4f}s</p>", unsafe_allow_html=True)
            
            if not df_delayed.empty:
                delay_count = len(df_delayed)
                delay_percent = (delay_count / total_orders) * 100
                avg_delay = df_delayed["Trễ hẹn (Ngày)"].mean()
                
                # Hiển thị Metric cảnh báo
                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng số đơn", f"{total_orders:,}")
                m2.metric("Đơn bị lùi lịch", f"{delay_count:,}", f"-{delay_percent:.1f}%", delta_color="inverse")
                m3.metric("Lùi trung bình", f"{avg_delay:.1f} ngày", "Vượt tải", delta_color="inverse")
                
                st.write("")
                with st.expander("⚠️ Chi tiết danh sách các đơn hàng bị lùi lịch", expanded=False):
                    st.dataframe(df_delayed, use_container_width=True, hide_index=True)
            else:
                st.success("**Hệ thống đủ công suất hoàn thành 100% đơn hàng đúng hạn.** Không phát hiện điểm nghẽn (bottleneck).")

        # --- TRA CỨU CHI TIẾT ---
        with st.container(border=True):
            st.markdown("<p class='chart-title'>Tra cứu lịch trình nhà máy</p>", unsafe_allow_html=True)
            st.markdown("<span class='chart-desc'>Tra cứu kế hoạch chạy máy cụ thể theo từng ngày và từng nhà máy.</span>", unsafe_allow_html=True)
            
            col_f, col_d = st.columns(2)
            with col_f:
                factory_options = [f"Nhà máy {i}" for i in range(1, k_facs + 1)]
                selected_factory_str = st.selectbox("Lọc theo nhà máy:", ["Tất cả"] + factory_options)
                
            with col_d:
                df_schedule["Date"] = pd.to_datetime(df_schedule["Date"]).dt.date
                min_date, max_date = df_schedule["Date"].min(), df_schedule["Date"].max()
                selected_date = st.date_input("Lọc theo ngày:", min_value=min_date, max_value=max_date, value=min_date)

            st.write("")
            df_lookup = df_schedule[df_schedule["Date"] == selected_date]
            if selected_factory_str != "Tất cả":
                fac_id = int(selected_factory_str.split(" ")[-1])
                df_lookup = df_lookup[df_lookup["Factory_ID"] == fac_id]
                
            if not df_lookup.empty:
                # Sửa lỗi ngoặc kép lồng nhau bằng cách dùng dấu nháy đơn '%' cho strftime
                st.markdown(f"**Danh sách các đơn hàng thi công ngày {selected_date.strftime('%d/%m/%Y')}**")
                df_display = df_lookup[["Factory_ID", "Order_ID", "Work_Percent"]].copy()
                df_display.columns = ["Nhà máy số", "Mã đơn hàng", "Công suất (%)"]
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("Không có kế hoạch sản xuất nào khớp với điều kiện tra cứu.")

        # --- TRÍCH XUẤT KẾ HOẠCH ---
        with st.container(border=True):
            st.markdown("<p class='chart-title'> Trích xuất kế hoạch vận hành</p>", unsafe_allow_html=True)
            
            df_export = format_export_data(df_schedule)
            st.dataframe(
                df_export.head(30), 
                height=210, 
                use_container_width=True,
                hide_index=True
            )
            
            st.write("")
            csv_schedule = df_export.to_csv(index=False).encode("utf-8")
            
            col_btn_export, _ = st.columns([1, 3])
            with col_btn_export:
                st.download_button(
                    label="Tải file kế hoạch (CSV)",
                    data=csv_schedule,
                    file_name="schedule_optimized.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )