# MODEL_RESULTS

Khi chạy thử nghiệm các mô hình deep learning, dù cho đã cài đặt seed ngẫu nhiên để đảm bảo tính tái lập (chi tiết xem tại [đây](../utils/set_up.py)), chúng tôi vẫn nhận thấy sự biến động nhẹ về kết quả giữa các lần chạy. Điều này có thể do nhiều yếu tố cụ thể:
- Khi chạy trên GPU, thư viện cuDNN của NVIDIA tối ưu hóa tốc độ bằng cách sử dụng các thao tác Atomic Add (Cộng nguyên tử) trên nhiều luồng (threads) chạy song song. Việc luồng nào tính xong trước và ghi dữ liệu vào bộ nhớ trước là **hoàn toàn ngẫu nhiên** tùy thuộc vào tình trạng phần cứng tại mili-giây đó.
- Mặc dù trong toán học lý tưởng $(A + B) + C = A + (B + C)$, nhưng trong tính toán dấu phẩy động (Floating-point) của máy tính, việc cộng theo các thứ tự khác nhau sẽ sinh ra sai số làm tròn siêu nhỏ (ví dụ lệch ở chữ số thập phân thứ 8). Qua hàng chục epochs và hàng nghìn tham số, các sai số siêu nhỏ này tích tụ lại (gọi là hiệu ứng cánh bướm) khiến kết quả F1 hoặc Loss cuối cùng bị lệch.

Vì vậy chúng tôi tiến hành chạy nhiều lần (4 lần) trên mỗi mô hình đề xuất và ghi lại kết quả tốt nhất (best), trung bình (average), độ lệch chuẩn (Variance) cũng như thời gian chạy để đảm bảo tính khách quan và minh bạch trong việc đánh giá hiệu suất của các mô hình.

Các mô hình baseline RNN, LSTM và GRU cũng được chạy 4 lần để so sánh công bằng với các mô hình đề xuất. Tuy nhiên, do kết quả trên tập Validation thấp cũng như giới hạn số lần nộp bài của cuộc thi nên các mô hình baseline sẽ không có kết quả trên tập Public Test.

**Lưu ý**: Thời gian chạy được đề cập trong file này là ước tính trung bình 4 lần chạy trên NVIDIA GPU Tesla T4 của Kaggle. Thời gian thực tế khi chạy có thể thay đổi, nhưng không vượt quá **1.5 lần** thời gian ước tính.

## TAANet: Temporal Actions Attention Network (54m46s)
| Metric | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 0.9989 | 0.9989 | 1.0000 | 1.0000 | 1.0000 | 0.9995 | 0.0006 |
|Macro F1 Attribute 2| 0.9999 | 0.9999 | 0.9994 | 0.9994 | 0.9999 | 0.9997 | 0.0001 |
|Macro F1 Attribute 3| 0.9983 | 0.9983 | 0.9980 | 0.9980 | 0.9983 | 0.9982 | 0.0001 |
|Macro F1 Attribute 4| 0.9928 | 0.9928 | 0.9927 | 0.9927 | 0.9928 | 0.9928 | 0.0001 |
|Macro F1 Attribute 5| 0.9993 | 0.9993 | 0.9994 | 0.9994 | 0.9994 | 0.9993 | 0.0001 |
|Macro F1 Attribute 6| 0.9991 | 0.9991 | 0.9992 | 0.9992 | 0.9992 | 0.9991 | 0.0001 |
|Validation Exact Match| 0.9968 | 0.9968 | 0.9967 | 0.9967 | 0.9968 | 0.9967 | 0.0001 |
|Public Test Exact Match| 0.9574 | 0.9574 | 0.9579 | 0.9579 | 0.9579 | 0.9577 | 0.0003 |

## TACNet: Temporal Actions Convolutional Network (35m50s)
| Metric | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 2| 1.0000 | 1.0000 | 0.9999 | 0.9999 | 1.0000 | 0.9999 | 0.0001 |
|Macro F1 Attribute 3| 0.9987 | 0.9986 | 0.9982 | 0.9982 | 0.9987 | 0.9984 | 0.0002 |
|Macro F1 Attribute 4| 0.9890 | 0.9924 | 0.9944 | 0.9972 | 0.9972 | 0.9932 | 0.0008 |
|Macro F1 Attribute 5| 0.9993 | 0.9985 | 0.9993 | 0.9993 | 0.9993 | 0.9991 | 0.0004 |
|Macro F1 Attribute 6| 0.9993 | 0.9989 | 0.9991 | 0.9991 | 0.9993 | 0.9991 | 0.0002 |
|Validation Exact Match| 0.9992 | 0.9968 | 0.9972 | 0.9969 | 0.9992 | 0.9975 | 0.0011 |
|Public Test Exact Match| 0.9675 | 0.9662 | 0.9581 | 0.9735 | 0.9735 | 0.9663 | 0.0061 |

## TARNet: Temporal Actions Recurrent Network (22m37s)
| Metric | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 2| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 3| 0.9987 | 0.9988 | 0.9987 | 0.9986 | 0.9988 | 0.9987 | 0.0001 |
|Macro F1 Attribute 4| 0.9972 | 0.9934 | 0.9945 | 0.9972 | 0.9972 | 0.9956 | 0.0017 |
|Macro F1 Attribute 5| 0.9993 | 0.9991 | 0.9993 | 0.9994 | 0.9994 | 0.9993 | 0.0001 |
|Macro F1 Attribute 6| 0.9992 | 0.9992 | 0.9989 | 0.9992 | 0.9992 | 0.9991 | 0.0001 |
|Validation Exact Match| 0.9976 | 0.9975 | 0.9974 | 0.9976 | 0.9976 | 0.9975 | 0.0001 |
|Public Test Exact Match| 0.9654 | 0.9703 | 0.9746 | 0.9703 | 0.9746 | 0.9701 | 0.0033 |

## TAGNet: Temporal Actions Graph Network (30m28s)
| Metric | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 2| 1.0000 | 0.9998 | 1.0000 | 0.9998 | 1.0000 | 0.9999 | 0.0001 |
|Macro F1 Attribute 3| 0.9983 | 0.9984 | 0.9979 | 0.9982 | 0.9984 | 0.9982 | 0.0002 |
|Macro F1 Attribute 4| 0.9950 | 0.9949 | 0.9972 | 0.9949 | 0.9972 | 0.9958 | 0.0011 |
|Macro F1 Attribute 5| 0.9994 | 0.9994 | 0.9993 | 0.9993 | 0.9994 | 0.9993 | 0.0001 |
|Macro F1 Attribute 6| 0.9992 | 0.9989 | 0.9992 | 0.9990 | 0.9992 | 0.9991 | 0.0002 |
|Validation Exact Match| 0.9974 | 0.9971 | 0.9969 | 0.9968 | 0.9974 | 0.9971 | 0.0002 |
|Public Test Exact Match| 0.9692 | 0.9704 | 0.9725 | 0.9720 | 0.9725 | 0.9708 | 0.0013 |

## RNN Baseline (4m41s)
| Metric | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 0.8253 | 0.7997 | 0.7962 | 0.7953 | 0.8253 | 0.8041 | 0.0012 |
|Macro F1 Attribute 2| 0.3843 | 0.5445 | 0.5024 | 0.5548 | 0.5548 | 0.4965 | 0.0082 |
|Macro F1 Attribute 3| 0.0199 | 0.0696 | 0.0536 | 0.0357 | 0.0696 | 0.0447 | 0.0008 |
|Macro F1 Attribute 4| 0.7063 | 0.7512 | 0.7242 | 0.7584 | 0.7584 | 0.7100 | 0.0018 |
|Macro F1 Attribute 5| 0.1718 | 0.3001 | 0.3057 | 0.3622 | 0.3622 | 0.2849 | 0.0081 |
|Macro F1 Attribute 6| 0.0056 | 0.0095 | 0.0123 | 0.0072 | 0.0123 | 0.0087 | 0.0001 |
|Validation Exact Match| 0.0006 | 0.0014 | 0.0011 | 0.0007 | 0.0014 | 0.0009 | 0.0001 |

## LSTM Baseline (5m)
| Metric | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 0.8712 | 0.8585 | 0.8117 | 0.9064 | 0.9064 | 0.8619 | 0.0028 |
|Macro F1 Attribute 2| 0.9640 | 0.9529 | 0.9518 | 0.9582 | 0.9640 | 0.9557 | 0.0005 |
|Macro F1 Attribute 3| 0.0293 | 0.0469 | 0.0506 | 0.0469 | 0.0506 | 0.0434 | 0.0003 |
|Macro F1 Attribute 4| 0.8277 | 0.8234 | 0.7887 | 0.8125 | 0.8277 | 0.8131 | 0.0008 |
|Macro F1 Attribute 5| 0.7463 | 0.7017 | 0.7381 | 0.7537 | 0.7537 | 0.7349 | 0.0008 |
|Macro F1 Attribute 6| 0.0058 | 0.0084 | 0.0107 | 0.0076 | 0.0107 | 0.0081 | 0.0001 |
|Validation Exact Match| 0.0010 | 0.0008 | 0.0008 | 0.0011 | 0.0011 | 0.0009 | 0.0001 |

## GRU Baseline (5m53)
| Metric | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 0.9011 | 0.9280 | 0.8948 | 0.9185 | 0.9280 | 0.9106 | 0.0017 |
|Macro F1 Attribute 2| 0.9735 | 0.9685 | 0.9759 | 0.9734 | 0.9759 | 0.9728 | 0.0003 |
|Macro F1 Attribute 3| 0.1645 | 0.2939 | 0.3482 | 0.1739 | 0.3482 | 0.2451 | 0.0028 |
|Macro F1 Attribute 4| 0.8590 | 0.8551 | 0.8702 | 0.9067 | 0.9067 | 0.8728 | 0.0018 |
|Macro F1 Attribute 5| 0.8820 | 0.8562 | 0.8770 | 0.8768 | 0.8820 | 0.8725 | 0.0007 |
|Macro F1 Attribute 6| 0.0290 | 0.0590 | 0.0486 | 0.0360 | 0.0590 | 0.0432 | 0.0008 |
|Validation Exact Match| 0.0079 | 0.0231 | 0.0238 | 0.0117 | 0.0238 | 0.0166 | 0.0007 |
