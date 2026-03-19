import streamlit as st
import web.data_page
import web.infer_page
import web.scheduler_page

st.set_page_config(page_title="HD4K - DataFlow 2026", layout="wide")

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
        /* Ép sử dụng System Font của các HĐH để đảm bảo độ chuyên nghiệp */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Làm mờ Header của Streamlit */
    [data-testid="stHeader"] {
        background-color: rgba(3, 6, 16, 0.5) !important;
        backdrop-filter: blur(12px) !important;
    }
    
    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(30, 41, 59, 0.5);
    }
    
    /* Đồng bộ Font cho toàn bộ Sidebar */
    [data-testid="stSidebar"] * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* CSS riêng cho Trang chủ (Landing Page) */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 500;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .intro-text {
        font-size: 1.05rem;
        color: #CBD5E1;
        line-height: 1.7;
        text-align: center;
        max-width: 900px;
        margin: 0 auto 2.5rem auto;
    }
    
    /* Box Điểm sáng kiến trúc thay cho st.info */
    .highlight-box {
        background-color: rgba(30, 58, 138, 0.1);
        border-left: 4px solid #3B82F6;
        padding: 1.5rem 2rem;
        border-radius: 4px 8px 8px 4px;
        color: #E2E8F0;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 3rem;
    }

    /* Thẻ tính năng (Feature Cards) */
    .feature-card {
        background-color: #0c1223;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.8rem 1.5rem;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.1);
        border-color: #3B82F6;
    }
    .feature-title {
        color: #F8FAFC;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.8rem;
    }
    .feature-desc {
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Custom Scrollbar (Thanh cuộn tàng hình) */
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

    /* Hiệu ứng Mượt mà (Fade-in Animation) khi load component */
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(15px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    /* Áp dụng animation cho khối chứa nội dung chính */
    [data-testid="stMainBlockContainer"] > div {
        animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* 6. Nổi bật thanh Nav Sidebar đang được chọn */
    .stRadio [role="radiogroup"] label {
        transition: all 0.2s ease;
        padding: 4px 8px;
        border-radius: 6px;
    }
    .stRadio [role="radiogroup"] label:hover {
        background-color: rgba(30, 58, 138, 0.3);
        transform: translateX(5px); /* Hiệu ứng đẩy chữ sang phải nhẹ khi hover */
    }
    </style>
""", unsafe_allow_html=True)

# Cache Data
web.data_page.preload_data()
web.infer_page.preload_inference()

def show_intro():
    st.markdown("<div class='hero-title'>ChronoLens</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Đội thi: HD4K | Vòng chung kết DataFlow 2026 - The Alchemy of Minds</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='intro-text'>
            <b>ChronoLens</b> là bước bứt phá công nghệ nhằm giải quyết bài toán Hồi quy đa mục tiêu (Multi-target Regression) có độ phức tạp cao nhất. <br><br>
            Mục tiêu cốt lõi không dừng lại ở việc phân loại hành vi tĩnh, mà là <b>dự báo thời gian giao dịch và công suất nhà máy</b> trực tiếp từ các chuỗi hành động biến thiên của khách hàng. Dự án đóng vai trò là chiếc cầu nối hoàn hảo giữa thế giới dữ liệu số và hoạt động vận hành vật lý, giúp doanh nghiệp chủ động điều tiết chuỗi cung ứng và lập kế hoạch sản xuất sát với thực tế.
        </div>
    """, unsafe_allow_html=True)

    col_space1, col_center, col_space2 = st.columns([1, 8, 1])
    with col_center:
        st.markdown("""
            <div class='highlight-box'>
                <b>Điểm sáng tạo - Kiến trúc Hybrid Neural-ML:</b> Bằng cách tập trung các biểu diễn ngữ cảnh từ mạng Neural đồ thị đa nhiệm và kết hợp với sức mạnh dò tìm sai số vi mô của các mô hình cây quyết định, hệ thống không chỉ triệt tiêu các điểm mù của mạng học sâu mà còn đem lại độ chính xác cực hạn trên thước đo wMSE.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #E2E8F0; margin-bottom: 1.5rem; text-align: center; font-weight: 600;'>Hệ sinh thái ChronoLens</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-title'>Khai phá dữ liệu</div>
                <div class='feature-desc'>Trực quan hóa bức tranh toàn cảnh về chuỗi hành vi, giải mã các quy luật thời gian, tỷ lệ đa dạng hành động và những tín hiệu mỏ neo quyết định đến mức công suất.</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-title'>Dự đoán trực tiếp</div>
                <div class='feature-desc'>Trải nghiệm sức mạnh của Cỗ máy Neural-ML trong thời gian thực. Hệ thống tự động dịch thuật các chuỗi hành động hỗn độn thành 6 tham số vận hành chuẩn xác.</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-title'>Xây dựng lịch trình</div>
                <div class='feature-desc'>Ứng dụng trực tiếp kết quả dự báo vào bài toán tối ưu hóa nguồn lực. Mô phỏng cách doanh nghiệp phân bổ công suất nhà máy tự động dựa trên dòng thời gian được dự đoán.</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown("<p style='text-align: center; color: #475569; font-size: 0.9rem;'>© 2026 Team HD4K. Built for DataFlow 2026.</p>", unsafe_allow_html=True)

def main():
    menu_options = [
        "Trang chủ", 
        "Khám phá dữ liệu", 
        "Dự đoán và Giải thích",
        "Lịch trình và Kịch bản" 
    ]
    
    # Custom Sidebar Header
    st.sidebar.markdown("<h2 style='color: #F8FAFC; text-align: center; margin-bottom: 1rem; font-weight: 700;'>ChronoLens Menu</h2>", unsafe_allow_html=True)
    
    choice = st.sidebar.radio("", menu_options, label_visibility="collapsed")
    
    st.sidebar.divider()
    
    # Thông tin bổ sung ở Sidebar (Ký tự tròn ● là hình học chuẩn, không phải emoji)
    st.sidebar.markdown("""
        <div style='color: #64748B; font-size: 0.85rem; text-align: center;'>
            Trạng thái hệ thống: <span style='color: #10B981;'>● Online</span><br>
            Phiên bản: 2.0 (Final)
        </div>
    """, unsafe_allow_html=True)

    if choice == "Trang chủ":
        show_intro()
    elif choice == "Khám phá dữ liệu":
        web.data_page.show()
    elif choice == "Dự đoán và Giải thích":
        web.infer_page.show()
    elif choice == "Lịch trình và Kịch bản":
        web.scheduler_page.show()

if __name__ == "__main__":
    main()