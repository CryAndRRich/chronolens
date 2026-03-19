<div align="center">
    <h1>[DataFlow 2026 - HD4K - User Behavior Prediction] <br> TAGFlow: Decoding Customer Sequences <br> via Multi-Task Neural Networks</h1>
    
[![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/kaggle-20BEFF?logo=kaggle&logoColor=white)]()
[![Visual Studio](https://badgen.net/badge/icon/visualstudio?icon=visualstudio&label)](https://visualstudio.microsoft.com)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
</div>


## 📖 Giới thiệu

**TAGFlow** là giải pháp phân tích và dự báo chuỗi hành vi người dùng do đội thi **HD4K** phát triển, vinh dự góp mặt tại vòng bán kết cuộc thi **"DataFlow 2026: The Alchemy of Minds"**.

Dự án giải quyết bài toán phân loại đa nhiệm (Multi-output Classification) trong lĩnh vực thương mại điện tử. Mục tiêu cốt lõi là dự báo chính xác và đồng thời **6 thuộc tính độc lập** đại diện cho quyết định mua sắm của khách hàng, dựa trên các chuỗi hành động có độ dài biến thiên (variable-length sequences). Giải pháp này hỗ trợ trực tiếp cho bài toán tối ưu hóa chuỗi cung ứng, giúp doanh nghiệp chủ động phân bổ nguồn lực và quản lý kho bãi hiệu quả.

Điểm làm nên sự khác biệt của TAGFlow là việc phá vỡ giới hạn của các mô hình "hộp đen". Hệ thống không chỉ chinh phục thước đo khắt khe Exact-Match Accuracy thông qua kiến trúc Mạng nơ-ron Đồ thị đa nhiệm (**TAGNet**) và quy trình dán nhãn giả 2 giai đoạn, mà còn tích hợp sâu sắc tính giải thích (**Explainable AI**). TAGFlow có khả năng lập bản đồ luồng hành vi khách hàng, phát hiện các "nút thắt" điều hướng trọng yếu và lọc bỏ các tín hiệu gây nhiễu, giúp doanh nghiệp thấu hiểu sâu sắc động lực mua sắm thực sự phía sau những luồng click chuột.

Nếu bạn thấy dự án này hữu ích, hãy ủng hộ chúng tôi một ngôi sao ⭐ trên GitHub nhé!

## 📂 Cấu trúc Dự án
```text
tagflow/
├── data/
│   ├── submissions/                            # Các file kết quả nộp bài (submission)
│   └── weights/                                # Nơi lưu trữ trọng số (weights) của các mô hình
│
├── config/
│   ├── config_data.py                          # Thiết lập xử lý dữ liệu
│   └── config_model.py                         # Thiết lập cấu hình mô hình
│
├── preprocess/
│   ├── __init__.py                             # Lớp DataManager quản lý toàn bộ pipeline xử lý dữ liệu
│   ├── dataloader.py                           # DataLoader tùy chỉnh cho dữ liệu
│   └── preprocess_data.py                      # Tiền xử lý dữ liệu 
│
├── model/
│   ├── models/                             
│   │   ├── baselines/                          # Các mô hình cơ sở
│   │   │   ├── ml.py/                          # Nhóm mô hình học máy
│   │   │   └── rnn.py/                         # Nhóm mô hình mạng hồi quy
│   │   │
│   │   ├── attention.py                        # Module Attention1D tùy chỉnh
│   │   ├── taanet.py                           # Mô hình TAANet 
│   │   ├── tacnet.py                           # Mô hình TACNet
│   │   ├── tarnet.py                           # Mô hình TARNet
│   │   └── tagnet.py                           # Mô hình TAGNet (kiến trúc chính)
│   │
│   ├── MODEL_RESULTS.md                        # Tài liệu ghi chép kết quả chạy của mô hình
│   │
│   └── train/                                  # Script huấn luyện mô hình
│
├── explainer/                                  # Module giải thích mô hình TAGNet
│   ├── error_attn.py                           # Phân tích nhãn sai
│   ├── global_attn.py                          # Phân tích toàn cục
│   ├── graph_attn.py                           # Phân tích đồ thị
│   └── integrate_grad.py                       # Phân tích tích hợp gradient
│
├── utils/
│   ├── set_up.py                               # Thiết lập môi trường, đảm bảo tính tái lập
│   ├── evaluate.py                             # Các hàm tính toán metric đánh giá
│   ├── prepare_model.py                        # Hàm chuẩn bị mô hình, tải trọng số,...
│   └── plot_graph.py                           # Các hàm hỗ trợ vẽ đồ thị, biểu đồ,...
│
├── scripts/                                    
│   ├── dataflow2026_hd4k_insight.ipynb         # Script chạy phân tích dữ liệu
│   ├── dataflow2026_hd4k_run_pipeline.ipynb    # Script chạy toàn bộ pipeline
│   ├── dataflow2026_hd4k_run_baselines.ipynb   # Script chạy các mô hình cơ sở
│   ├── dataflow2026_hd4k_run_xai.ipynb         # Script giải thích mô hình với xAI
│   │
│   └── HOW_TO_RUN_KAGGLE.md                    # Hướng dẫn chạy trên Kaggle
│
├── report/                                    
│   ├── img/                                    # Ảnh sử dụng trong report, README
│   ├── TAGFlow_report.pdf                      # File báo cáo dự án
│   └── TAGFlow_slide_pdf.pdf                   # Slide thuyết trình dự án (pdf)
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