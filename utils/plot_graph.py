from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph_network(edges: Dict[Tuple[int, int], float], 
                       top_k: int = 100, 
                       max_occurrences: int = 10) -> None:
    """
    Vẽ đồ thị nhân quả
    """
    filtered_edges = []
    node_counts = defaultdict(int)
    
    for (src, tgt), weight in edges.items():
        if node_counts[src] < max_occurrences and node_counts[tgt] < max_occurrences:
            filtered_edges.append(((src, tgt), weight))
            
            # Cập nhật số lần xuất hiện
            node_counts[src] += 1
            node_counts[tgt] += 1
            
        if len(filtered_edges) >= top_k:
            break

    # Khởi tạo đồ thị có hướng (Directed Graph)
    G = nx.DiGraph()
    
    for (src, tgt), weight in filtered_edges:
        G.add_edge(src, tgt, weight=weight)
        
    # Setup khung vẽ
    plt.figure(figsize=(14, 10))
    
    # Sử dụng thuật toán spring_layout
    pos = nx.spring_layout(G, k=0.8, seed=42) 
    
    # Trích xuất thuộc tính để tinh chỉnh giao diện
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    min_w, max_w = min(edge_weights), max(edge_weights)
    
    # Chuẩn hóa độ dày cạnh (từ 0.5 đến 3.0)
    if max_w > min_w:
        edge_widths = [0.5 + 2.5 * ((w - min_w) / (max_w - min_w)) for w in edge_weights]
    else:
        edge_widths = [1.5] * len(edge_weights)
        
    # Tính toán In-degree để tô màu nút
    in_degrees = dict(G.in_degree())
    node_colors = [in_degrees[node] for node in G.nodes()]

    # Bắt đầu vẽ đồ thị
    _ = nx.draw_networkx_nodes(
        G, pos, 
        node_size=800,  
        node_color=node_colors, 
        cmap=plt.cm.YlOrRd, 
        edgecolors="black", 
        linewidths=1.0
    )
    
    edges = nx.draw_networkx_edges(
        G, pos, 
        arrowstyle="-|>", 
        arrowsize=15,
        width=edge_widths, 
        edge_color="dimgray",
        connectionstyle="arc3,rad=0.1"
    )
    
    nx.draw_networkx_labels(
        G, pos, 
        font_size=8,   
        font_weight="bold", 
        font_family="sans-serif"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_distractor_analysis(distractors: List[Tuple[int, float, float, float]],
                             top_k: int = 5):
    """
    Vẽ biểu đồ cột ghép so sánh Attention giữa nhóm đoán Sai và đoán Đúng.
    """
    # Trích xuất dữ liệu cho Top K
    actual_top = min(top_k, len(distractors))
    top_distractors = distractors[:actual_top]
    
    action_ids = [str(item[0]) for item in top_distractors]
    wrong_weights = [item[2] for item in top_distractors]
    correct_weights = [item[3] for item in top_distractors]
    diff_weights = [item[1] for item in top_distractors]

    x = np.arange(len(action_ids))  # Vị trí các nhãn
    width = 0.35  # Độ rộng của mỗi cột

    # Khởi tạo khung vẽ
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vẽ cột cho nhóm Sai (Màu Đỏ - cảnh báo) và Đúng (Màu Xanh - an toàn)
    rects1 = ax.bar(x - width/2, wrong_weights, width, label="Đoán Sai", color="crimson", edgecolor="black", alpha=0.8)
    rects2 = ax.bar(x + width/2, correct_weights, width, label="Đoán Đúng", color="mediumseagreen", edgecolor="black", alpha=0.8)

    ax.set_ylabel("Mức độ đóng góp (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mã hành động", fontsize=12, fontweight="bold")
    ax.set_title(f"Phân tích Top {actual_top} hành động gây nhiễu (Distractors)", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(action_ids, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Viết số % lên đầu mỗi cột
    def autolabel(rects: List[plt.Rectangle]) -> None:
        """
        Đính kèm text label lên đầu các cột
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # Dịch lên 1 points
                        textcoords="offset points",
                        ha="center", 
                        va="bottom", 
                        fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    for i in range(len(action_ids)):
        max_height = max(wrong_weights[i], correct_weights[i])
        ax.text(x[i], 
                max_height + 2.5, 
                rf"$\Delta$ +{diff_weights[i]:.1f}%", 
                ha="center", 
                va="bottom", 
                fontsize=10, 
                color="red", 
                fontweight="bold",
                bbox=dict(facecolor="white", 
                          edgecolor="red", 
                          boxstyle="round,pad=0.2", 
                          alpha=0.8
                ))

    absolute_max = max(max(wrong_weights), max(correct_weights))
    ax.set_ylim(0, absolute_max + 8)
    
    fig.tight_layout()
    plt.show()