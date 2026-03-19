<div align="center">
    <h1>[DataFlow 2026 - HD4K - User Behavior Prediction] <br> ChronoLens: Decoding Customer Behaviors via Neural <br> Representation and ML-Driven Web Frameworks</h1>
    
[![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/kaggle-20BEFF?logo=kaggle&logoColor=white)]()
[![Visual Studio](https://badgen.net/badge/icon/visualstudio?icon=visualstudio&label)](https://visualstudio.microsoft.com)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
</div>


## 📖 Giới thiệu

**ChronoLens** là giải pháp phân tích và dự báo chuỗi hành vi người dùng do đội thi **HD4K** phát triển, vinh dự góp mặt tại vòng chung kết cuộc thi **"DataFlow 2026: The Alchemy of Minds"** với mục tiêu giải quyết bài toán Hồi quy đa mục tiêu (Multi-target Regression).

Mục tiêu cốt lõi của ChronoLens không dừng lại ở việc phân loại hành vi tĩnh, mà là **dự báo thời gian giao dịch và công suất nhà máy** trực tiếp từ các chuỗi hành động biến thiên của khách hàng. Dự án đóng vai trò là chiếc cầu nối hoàn hảo giữa thế giới dữ liệu số và hoạt động vận hành vật lý, giúp doanh nghiệp chủ động điều tiết chuỗi cung ứng và lập kế hoạch sản xuất sát với thực tế.

Điểm sáng tạo làm nên vị thế của ChronoLens là kiến trúc **Hybrid Neural-ML**. Bằng cách "chưng cất" các biểu diễn ngữ cảnh từ mạng Neural đồ thị đa nhiệm và kết hợp với sức mạnh dò tìm sai số vi mô của các mô hình cây quyết định, hệ thống không chỉ triệt tiêu các điểm mù của mạng học sâu mà còn đem lại độ chính xác cực hạn trên thước đo WMSE.

ChronoLens gồm 3 phần chính:
- **Khám phá dữ liệu:** Trực quan hóa bức tranh toàn cảnh về chuỗi hành vi, giải mã các quy luật thời gian, tỷ lệ đa dạng hành động và những "tín hiệu mỏ neo" quyết định đến mức công suất.
- **Dự đoán trực tiếp:** Trải nghiệm sức mạnh của Cỗ máy Neural-ML trong thời gian thực. Hệ thống tự động dịch thuật các chuỗi hành đồng hỗn độn thành 6 tham số vận hành chuẩn xác.
- **Xây dựng lịch trình:** Ứng dụng trực tiếp kết quả dự báo vào bài toán tối ưu hóa nguồn lực. Mô phỏng cách doanh nghiệp phân bổ công suất nhà máy tự động dựa trên dòng thời gian được dự đoán.
    
Nếu bạn thấy dự án này hữu ích, hãy ủng hộ chúng tôi một ngôi sao ⭐ trên GitHub nhé!

## 📂 Cấu trúc Dự án
```text
chronolens/
├── data/
│   └── weights/                                # Nơi lưu trữ trọng số (weights) của các mô hình
│
├── config/
│   ├── config_data.py                          # Thiết lập xử lý dữ liệu
│   └── config_model.py                         # Thiết lập cấu hình mô hình
│
├── preprocess/
│   ├── __init__.py                             # Lớp DataManager quản lý toàn bộ pipeline xử lý dữ liệu
│   ├── dataloader.py                           # DataLoader tùy chỉnh cho dữ liệu
│   ├── embedding.py                            # Trích xuất embedding từ chuỗi hành động 
│   └── preprocess_data.py                      # Tiền xử lý dữ liệu 
│
├── model/
│   ├── chrono_net/                             
│   │   ├── hypertuning/                        # Tìm kiếm siêu tham số cho mô hình cây
│   │   │   ├── xgb.py/                       
│   │   │   └── lgbm.py/                        
│   │   │
│   │   ├── layers.py                           # Module Attention1D, GCEFusion và CascadeRegression
│   │   ├── chrono_c.py                         # Mô hình ChronoC (Convolutional)
│   │   ├── chrono_g.py                         # Mô hình ChronoG (Graph)
│   │   └── chrono_r.py                         # Mô hình ChronoR (Recurrent)
│   │
│   ├── MODEL_RESULTS.md                        # Tài liệu ghi chép kết quả chạy của mô hình
│   │
│   ├── loss.py                                 # Hàm loss tùy chỉnh
│   └── train.py                                # Script huấn luyện mô hình
│
├── explainer/                                  # Module giải thích mô hình
│   ├── error_attn.py                           # Phân tích nhãn sai
│   └── graph_attn.py                           # Phân tích đồ thị
│
├── utils/
│   ├── set_up.py                               # Thiết lập môi trường, đảm bảo tính tái lập
│   ├── evaluate.py                             # Các hàm tính toán metric đánh giá
│   ├── prepare_model.py                        # Hàm chuẩn bị mô hình, tải trọng số,...
│   └── plot_graph.py                           # Các hàm hỗ trợ vẽ đồ thị, biểu đồ,...
│
├── scripts/                                    
│   ├── dataflow2026_hd4k_insight.ipynb         # Script chạy phân tích dữ liệu
│   ├── dataflow2026_hd4k_run_stage_1.ipynb     # Script chạy toàn bộ pipeline
│   ├── dataflow2026_hd4k_run_stage_2.ipynb     # Script chạy các mô hình cơ sở
│   ├── dataflow2026_hd4k_run_xai.ipynb         # Script giải thích mô hình với xAI
│   │
│   └── HOW_TO_RUN_KAGGLE.md                    # Hướng dẫn chạy trên Kaggle
│
├── report/                                    
│   ├── img/                                    # Ảnh sử dụng trong report, README
│   ├── ChronoLens_report.pdf                   # File báo cáo dự án
│   └── ChronoLens_slide_pdf.pdf                # Slide thuyết trình dự án (pdf)
│
├── .gitignore                       
├── LICENSE                                     # Giấy phép MIT
├── requirements.txt                            # Danh sách thư viện cần thiết
└── README.md                                      
```

## 💻 Yêu cầu Hệ thống & Hướng dẫn Sử dụng
Có tổng tất cả 4 scripts, tất cả đều cần chạy trên Kaggle, cụ thể:
- Script chạy phân tích dữ liệu và rút ra insight: dataflow2026_hd4k_insight.ipynb  
- Script chạy toàn bộ pipeline huấn luyện mô hình: dataflow2026_hd4k_run_baselines.ipynb
- Script chạy các mô hình cơ sở để so sánh: dataflow2026_hd4k_run_explainer.ipynb
- Script giải thích mô hình với xAI: dataflow2026_hd4k_run_xai.ipynb

Chi tiết thông tin, hướng dẫn và thời gian chạy từng script có thể đọc trong chính các file jupyter notebook.

## 📜 Giấy phép
Dự án được phân phối dưới giấy phép MIT. Xem file [LICENSE](LICENSE) để biết chi tiết.

## 📞 Liên hệ
Mọi thắc mắc hoặc góp ý, xin vui lòng liên hệ với chúng tôi qua GitHub Issues, LinkedIn hoặc Facebook:

[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/CryAndRRich/trustee)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/in/cryandrich/)
[![Facebook](https://img.shields.io/badge/Facebook-0866FF?style=flat&logo=facebook&logoColor=white)](https://www.facebook.com/namhai.tran.73550794)

Chúng tôi trân trọng mọi phản hồi và đóng góp của bạn để giúp dự án ngày càng hoàn thiện hơn!