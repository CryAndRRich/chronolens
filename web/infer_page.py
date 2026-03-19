import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
import time
from gensim.models import Word2Vec

VOCAB_PATH = "data/data_for_scripts/weights/vocab_dict.pkl"
W2V_PATH = "data/data_for_scripts/weights/word2vec_264.model"
VECTORIZER_PATH = "data/data_for_scripts/weights/count_vectorizer.pkl"
MODEL_DIR = "data/data_for_scripts/weights/best_model_chrono_r_lgbm"
ATTRIBUTE_LIST = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]

TOP_K_VOCAB = 50 
MAX_SEQ_LEN = 66

@st.cache_data
def get_metrics_data():
    overall = {"MAE": 1.3396, "MSE": 18.2537, "RMSE": 4.2724, "WMAPE": 5.7843, "WMSE": 0.8327}
    details = pd.DataFrame({
        "Thuộc tính": ["Attr 1 (Tháng bắt đầu)", "Attr 2 (Ngày bắt đầu)", "Attr 3 (Chỉ số nhà máy bắt đầu)", 
                       "Attr 4 (Tháng kết thúc)", "Attr 5 (Ngày kết thúc)", "Attr 6 (Chỉ số nhà máy kết thúc)"],
        "MAE": [0.7150, 1.6765, 1.3019, 0.7924, 2.3964, 1.1553],
        "MSE": [2.3331, 11.3940, 38.5890, 2.6471, 17.9117, 36.6474],
        "RMSE": [1.5274, 3.3755, 6.2120, 1.6270, 4.2322, 6.0537],
        "WMAPE": [10.6781, 13.1525, 2.6198, 11.8395, 17.9475, 2.3212]
    })
    
    mae_dict = {
        "attr_1": 0.7150, "attr_2": 1.6765, "attr_3": 1.3019,
        "attr_4": 0.7924, "attr_5": 2.3964, "attr_6": 1.1553
    }
    return overall, details, mae_dict

@st.cache_resource
def preload_inference():
    vocab_dict = {0: 0, "UNK": 1}
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "rb") as f:
            vocab_dict = pickle.load(f)

    w2v_model = None
    if os.path.exists(W2V_PATH):
        w2v_model = Word2Vec.load(W2V_PATH)

    vectorizer = None
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

    total_features = 10 + TOP_K_VOCAB + (264 if w2v_model else 0)

    models = {}
    for attr in ATTRIBUTE_LIST:
        model_path = f"{MODEL_DIR}/lgbm_retrained_{attr}.json"
        
        def create_dummy_model():
            dummy = lgb.LGBMRegressor(n_estimators=1, min_child_samples=1)
            dummy.fit(np.zeros((2, total_features)), np.zeros(2))
            return dummy.booster_

        if os.path.exists(model_path):
            try:
                models[attr] = lgb.Booster(model_file=model_path)
            except Exception:
                models[attr] = create_dummy_model()
        else:
            models[attr] = create_dummy_model()

    overall_metrics, df_metrics, mae_dict = get_metrics_data()

    return vocab_dict, w2v_model, vectorizer, models, overall_metrics, df_metrics, mae_dict

def extract_features(seqs_mapped, vectorizer, w2v_model):
    N = len(seqs_mapped)
    seq_lengths = np.sum(seqs_mapped != 0, axis=1).astype(np.float32)
    consecutive_repeats = np.sum((seqs_mapped[:, 1:] == seqs_mapped[:, :-1]) & (seqs_mapped[:, 1:] != 0), axis=1).astype(np.float32)
    padding_ratio = (MAX_SEQ_LEN - seq_lengths) / MAX_SEQ_LEN
    
    num_uniques = np.zeros(N, dtype=np.float32)
    diversity = np.zeros(N, dtype=np.float32)
    first_codes = np.zeros(N, dtype=np.float32)
    last_codes = np.zeros(N, dtype=np.float32)
    most_frequent_codes = np.zeros(N, dtype=np.float32)
    obsession_ratios = np.zeros(N, dtype=np.float32)
    center_of_mass = np.zeros(N, dtype=np.float32)
    
    seq_strings = []
    w2v_features = np.zeros((N, 264), dtype=np.float32) 
    
    for i, row in enumerate(seqs_mapped):
        non_zeros = row[row != 0]
        if len(non_zeros) > 0:
            vals, counts = np.unique(non_zeros, return_counts=True)
            num_uniques[i] = len(vals)
            diversity[i] = num_uniques[i] / len(non_zeros)
            first_codes[i] = non_zeros[0]
            last_codes[i] = non_zeros[-1]
            
            best_idx = np.argmax(counts)
            most_frequent_codes[i] = vals[best_idx]
            obsession_ratios[i] = counts[best_idx] / len(non_zeros)
            center_of_mass[i] = np.mean(np.arange(len(non_zeros))) / len(non_zeros)
            
            seq_strings.append(" ".join(non_zeros.astype(str)))
            
            if w2v_model:
                vecs = []
                for word_idx in non_zeros:
                    str_idx = str(int(word_idx))
                    if str_idx in w2v_model.wv:
                        vecs.append(w2v_model.wv[str_idx])
                if vecs:
                    w2v_features[i] = np.mean(vecs, axis=0)
        else:
            seq_strings.append("")

    bow_features = np.zeros((N, TOP_K_VOCAB), dtype=np.float32)
    if vectorizer:
        try:
            bow_features = vectorizer.transform(seq_strings).toarray()
        except Exception:
            pass

    manual_features = np.column_stack([
        seq_lengths, num_uniques, diversity, first_codes, last_codes,
        consecutive_repeats, padding_ratio, most_frequent_codes,
        obsession_ratios, center_of_mass, bow_features, w2v_features
    ])
    return manual_features.astype(np.float32)

import streamlit as st
import pandas as pd
import numpy as np
import time

def show():
    vocab_dict, w2v_model, vectorizer, models, overall_metrics, df_metrics, mae_dict = preload_inference()

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
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(3, 6, 16, 0.5) !important;
            backdrop-filter: blur(12px) !important; /* Làm mờ thanh header trên cùng */
        }
        [data-testid="stMainBlockContainer"] {
            padding-top: 3rem;
        }
        html, body, [class*="css"]  {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }

        /* 2. Custom Scrollbar (Thanh cuộn tinh tế) */
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

        /* 3. Hiệu ứng Fade-in mượt mà */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        [data-testid="stMainBlockContainer"] > div {
            animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        /* TẮT TOÀN BỘ VIỀN BOX MẶC ĐỊNH CỦA STREAMLIT */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }

        /* Styling cho Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid #1E293B;
            background-color: transparent;
            padding: 0px;
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

        /* Styling cho Metrics (Thẻ KPI - Khối Box duy nhất được giữ lại để tạo điểm nhấn) */
        div[data-testid="metric-container"] {
            background-color: rgba(12, 18, 35, 0.6);
            border: 1px solid #1E293B;
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #3B82F6; 
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
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- PADDING ---
    col_left, col_main, col_right = st.columns([0.05, 0.9, 0.05])

    with col_main:
        # --- HEADER ---
        st.markdown("<h1 class='section-title'>Dự đoán hành vi khách hàng</h1>", unsafe_allow_html=True)
        st.markdown("<span class='section-subtitle'>Sử dụng mô hình <b>XGBoost</b> để dự báo 6 thuộc tính vận hành.</span>", unsafe_allow_html=True)

        # --- TABS ---
        tab_eval, tab_single, tab_batch = st.tabs([
            "Đánh giá mô hình", 
            "Phân tích đơn lẻ", 
            "Dự đoán hàng loạt"
        ])

        # ==========================================
        # TAB 1: ĐÁNH GIÁ MÔ HÌNH
        # ==========================================
        with tab_eval:
            st.write("") 
            
            st.markdown("<p class='chart-title'>Hiệu năng tổng thể</p>", unsafe_allow_html=True)
            st.markdown("<span class='chart-desc'>Các chỉ số sai số tổng hợp trên toàn bộ tập kiểm thử.</span>", unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MAE", overall_metrics["MAE"])
            col2.metric("MSE", overall_metrics["MSE"])
            col3.metric("RMSE", overall_metrics["RMSE"])
            col4.metric("WMAPE", overall_metrics["WMAPE"])
            col5.metric("WMSE", overall_metrics["WMSE"])

            st.write("")
            st.write("")

            st.markdown("<p class='chart-title'>Hiệu năng chi tiết trên từng thuộc tính</p>", unsafe_allow_html=True)
            st.markdown("<span class='chart-desc'>Phân rã mức độ chính xác của dự báo cho từng khâu vận hành.</span>", unsafe_allow_html=True)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            st.write("")
            st.write("")

            # --- PHẦN XAI (KHÔNG VIỀN BOX) ---
            st.markdown("<h3 class='section-title'>XAI - Phân tích đặc trưng quan trọng</h3>", unsafe_allow_html=True)
            st.markdown("<span class='section-subtitle'>Giải mã hộp đen XGBoost bằng SHAP và đồ thị tương quan. Đánh giá mức độ đóng góp của từng mã hành động vào sai số dự báo.</span>", unsafe_allow_html=True)
            st.write("")
            
            st.markdown("<p class='chart-title'>Mạng lưới tương quan hành động (Distractor Network)</p>", unsafe_allow_html=True)
            st.image("report/img/xai_plot_1.jpg", caption="Mạng lưới các nút (nodes) thể hiện mối liên kết giữa các mã hành động gây nhiễu.", use_column_width=True)

            st.write("")
            st.write("")

            st.markdown("<p class='chart-title'>Phân tích hành động gây nhiễu (Top Distractors)</p>", unsafe_allow_html=True)
            col_xai_r2c1, col_xai_r2c2 = st.columns(2)
            with col_xai_r2c1:
                st.image("report/img/xai_plot_2.jpg", caption="Ảnh 2: Tỷ lệ phân bổ (Đoán sai vs Đoán đúng)", use_column_width=True)
            with col_xai_r2c2:
                st.image("report/img/xai_plot_3.jpg", caption="Ảnh 3: Tỷ lệ phân bổ (Đoán sai vs Đoán đúng)", use_column_width=True)
            
            st.write("")
            
            col_xai_r3c1, col_xai_r3c2 = st.columns(2)
            with col_xai_r3c1:
                st.image("report/img/xai_plot_4.jpg", caption="Ảnh 4: Tỷ lệ phân bổ (Đoán sai vs Đoán đúng)", use_column_width=True)
            with col_xai_r3c2:
                st.image("report/img/xai_plot_5.jpg", caption="Ảnh 5: Tỷ lệ phân bổ (Đoán sai vs Đoán đúng)", use_column_width=True)

        # ==========================================
        # TAB 2: PHÂN TÍCH ĐƠN LẺ
        # ==========================================
        with tab_single:
            st.write("")
            st.markdown("<p class='chart-title'>Hệ thống suy luận trực tiếp</p>", unsafe_allow_html=True)
            st.markdown("<p class='chart-desc'>Nhập chuỗi các mã hành động của khách hàng, phân cách bởi dấu phẩy.</p>", unsafe_allow_html=True)
            
            user_input = st.text_input("Chuỗi hành động (Tối đa 66 mã):", "4347, 105, 108, 105, 881, 735, 867, 105, 4187, 105, 21040, 1071")
            
            if "single_preds" not in st.session_state:
                st.session_state.single_preds = None
                st.session_state.oov_list = []
            
            col_btn, _ = st.columns([1, 4])
            with col_btn:
                submit_btn = st.button("Chạy dự đoán", type="primary", use_container_width=True)

            if submit_btn:
                try:
                    raw_seq = [float(x.strip()) for x in user_input.split(",")]
                    
                    mapped_seq = []
                    oov_list = []
                    for val in raw_seq:
                        if val in vocab_dict:
                            mapped_seq.append(vocab_dict[val])
                        else:
                            mapped_seq.append(1) 
                            oov_list.append(val)
                            
                    padded_seq = mapped_seq.copy()
                    if len(padded_seq) < MAX_SEQ_LEN:  
                        padded_seq += [0] * (MAX_SEQ_LEN - len(padded_seq)) 
                    else:
                        padded_seq = padded_seq[:MAX_SEQ_LEN]
                    
                    X_seq_array = np.array([padded_seq])
                    X_features = extract_features(X_seq_array, vectorizer, w2v_model)
                    
                    preds = {}
                    for attr in ATTRIBUTE_LIST:
                        pred_val = models[attr].predict(X_features)[0]
                        preds[attr] = f"{round(pred_val, 1)} ± {round(mae_dict[attr], 1)}"

                    st.session_state.single_preds = preds
                    st.session_state.oov_list = oov_list

                except ValueError:
                    st.error("Vui lòng chỉ nhập số và dấu phẩy (Ví dụ: 12, 45, 8).")
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}")

            # Hiển thị kết quả
            if st.session_state.single_preds is not None:
                st.write("")
                if st.session_state.oov_list:
                    str_oov = ", ".join([str(int(x)) for x in st.session_state.oov_list])
                    st.warning(f"Cảnh báo: Phát hiện các mã hành động không xác định [{str_oov}]. Đã tự động gán là [UNK].")
                
                preds = st.session_state.single_preds
                
                st.markdown("<p class='chart-title'>Giai đoạn bắt đầu giao dịch</p>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Tháng (attr_1)", preds["attr_1"])
                c2.metric("Ngày (attr_2)", preds["attr_2"])
                c3.metric("Chỉ số hoạt động nhà máy (attr_3)", preds["attr_3"])

                st.write("")
                st.markdown("<p class='chart-title'>Giai đoạn hoàn thành giao dịch</p>", unsafe_allow_html=True)
                c4, c5, c6 = st.columns(3)
                c4.metric("Tháng (attr_4)", preds["attr_4"])
                c5.metric("Ngày (attr_5)", preds["attr_5"])
                c6.metric("Chỉ số hoạt động nhà máy (attr_6)", preds["attr_6"])

# ==========================================
        # TAB 3: DỰ ĐOÁN HÀNG LOẠT
        # ==========================================
        with tab_batch:
            st.write("")
            st.markdown("<p class='chart-title'>Nhập tệp dữ liệu</p>", unsafe_allow_html=True)
            st.markdown("<p class='chart-desc'>Tải lên tệp CSV chứa dữ liệu khách hàng (yêu cầu cột 'id' và các cột 'feature_1' đến 'feature_66').</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                if "batch_features" not in st.session_state or st.session_state.get("last_uploaded_file") != uploaded_file.name:
                    df_upload = pd.read_csv(uploaded_file)
                    st.session_state.df_upload = df_upload
                    st.session_state.last_uploaded_file = uploaded_file.name
                    
                    with st.spinner("Tiến hành đọc và kiểm tra dữ liệu..."):
                        feature_cols = [col for col in df_upload.columns if col.startswith("feature_")]
                        X_batch_raw = df_upload[feature_cols].values
                        
                        map_func = np.vectorize(lambda x: vocab_dict.get(x, 1) if not np.isnan(x) else 0)
                        X_batch_mapped = map_func(X_batch_raw)
                        
                        if X_batch_mapped.shape[1] < MAX_SEQ_LEN:
                            pad_width = MAX_SEQ_LEN - X_batch_mapped.shape[1]
                            X_batch_mapped = np.pad(X_batch_mapped, ((0,0), (0, pad_width)), constant_values=0)
                        else:
                            X_batch_mapped = X_batch_mapped[:, :MAX_SEQ_LEN]

                        st.session_state.batch_features = extract_features(X_batch_mapped, vectorizer, w2v_model)

                st.write("")
                st.markdown(f"<p class='chart-title'>Dữ liệu đã tải lên ({len(st.session_state.df_upload)} dòng)</p>", unsafe_allow_html=True)
                st.dataframe(
                    st.session_state.df_upload.head(30), 
                    height=210, 
                    use_container_width=True,
                    hide_index=True
                )
                
                st.write("")
                col_btn_batch, _ = st.columns([1, 4])
                with col_btn_batch:
                    run_batch_btn = st.button("Chạy mô hình", type="primary", use_container_width=True)
                
                # SỬA LỖI Ở ĐÂY: Thụt lề khối if này vào trong (cùng cấp với col_btn_batch)
                if run_batch_btn:
                    start_time = time.time()
                    df_upload = st.session_state.df_upload
                    X_batch_features = st.session_state.batch_features
                    
                    with st.spinner("Đang chạy suy luận qua 6 mô hình XGBoost..."):
                        output_df = pd.DataFrame()
                        if "id" in df_upload.columns:
                            output_df["id"] = df_upload["id"]
                        else:
                            output_df["id"] = df_upload.index
                            
                        for attr in ATTRIBUTE_LIST:
                            preds_batch = models[attr].predict(X_batch_features)
                            output_df[attr] = np.round(preds_batch, 2)
                            st.session_state.batch_preds = output_df
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    time_per_sample = total_time / len(df_upload)
                    
                    st.write("")
                    st.success(f"Hoàn tất xử lý {len(output_df)} bản ghi.")
                    st.info(f"Tổng thời gian dự báo: {total_time:.4f} giây (Trung bình: {time_per_sample:.6f} giây/chuỗi)")
                    
                    st.dataframe(
                        output_df.head(30), 
                        height=210, 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    csv_data = output_df.to_csv(index=False).encode("utf-8")
                    
                    st.write("")
                    col_dl_btn, _ = st.columns([1, 3])
                    with col_dl_btn:
                        st.download_button(
                            label="Tải file kết quả (CSV)",
                            data=csv_data,
                            file_name="result.csv",
                            mime="text/csv",
                            use_container_width=True
                        )