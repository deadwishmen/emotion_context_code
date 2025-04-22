import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

def predict_and_show(models, device, data_loader, num_samples=5, class_names=None, conbine=False):
    """
    Dự đoán nhãn cho một số mẫu từ data_loader và hiển thị ảnh với nhãn dự đoán và nhãn thực.
    
    Args:
        models: Tuple chứa (model_context, model_body, model_face, model_text, fusion_model)
        device: Thiết bị để chạy mô hình (CPU hoặc GPU)
        data_loader: DataLoader chứa dữ liệu kiểm tra
        num_samples: Số lượng mẫu cần hiển thị
        class_names: Danh sách tên các lớp (26 lớp cảm xúc). Nếu None, sử dụng chỉ số lớp.
        conbine: Nếu là 'q_former', xử lý đặc biệt cho pred_context và pred_text
    """
    # Validate num_samples
    num_samples = min(num_samples, 10)  # Cap at 10 to avoid large figures
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    model_context, model_body, model_face, model_text, fusion_model = models
    
    # Chuyển các mô hình sang device và đặt ở chế độ đánh giá
    model_context.to(device).eval()
    model_body.to(device).eval()
    model_face.to(device).eval()
    model_text.to(device).eval()
    fusion_model.to(device).eval()
    
    # Nếu không có class_names, sử dụng chỉ số lớp
    if class_names is None:
        class_names = [f"Class {i}" for i in range(26)]
    
    # Tạo figure với kích thước hợp lý và DPI thấp hơn
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples), dpi=80)
    if num_samples == 1:
        axes = [axes]  # Đảm bảo axes là danh sách
    
    # Lưu trữ các mẫu để hiển thị
    displayed_samples = 0
    
    with torch.no_grad():
        for images_context, images_body, images_face, tokenizer_text, labels_cat, labels_cont in tqdm(iter(data_loader), total=len(data_loader)):
            # Chuyển dữ liệu sang device
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
            tokenizer_text = {key: val.to(device) for key, val in tokenizer_text.items()}
            labels_cat = labels_cat.to(device)
            
            # Dự đoán
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            if conbine == "q_former":
                pred_text = model_text(**tokenizer_text).last_hidden_state
                pred_context = model_context(images_context)[:, 1:]
            else:
                pred_text = model_text(**tokenizer_text).last_hidden_state.mean(dim=1)
                pred_context = model_context(images_context)
            pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)
            
            # Áp dụng sigmoid để chuyển logits thành xác suất
            pred_cat = torch.sigmoid(pred_cat).cpu().numpy()
            labels_cat = labels_cat.cpu().numpy()
            
            # Xử lý từng mẫu trong batch
            for i in range(images_context.shape[0]):
                if displayed_samples >= num_samples:
                    break
                
                # Lấy ảnh context (giả sử ảnh RGB, shape [C, H, W])
                img = images_context[i].cpu().numpy().transpose(1, 2, 0)  # Chuyển sang [H, W, C]
                img = (img - img.min()) / (img.max() - img.min())  # Chuẩn hóa để hiển thị
                
                # Lấy nhãn dự đoán và nhãn thực
                pred_labels = np.where(pred_cat[i] > 0.5)[0]  # Nhãn có xác suất > 0.5
                true_labels = np.where(labels_cat[i] == 1)[0]  # Nhãn thực
                
                # Chuyển chỉ số lớp thành tên lớp
                pred_label_names = [class_names[idx] for idx in pred_labels]
                true_label_names = [class_names[idx] for idx in true_labels]
                
                # Hiển thị ảnh và nhãn
                axes[displayed_samples].imshow(img)
                axes[displayed_samples].set_title(
                    f"Pred: {', '.join(pred_label_names) if pred_label_names else 'None'}\n"
                    f"True: {', '.join(true_label_names) if true_label_names else 'None'}",
                    fontsize=10
                )
                axes[displayed_samples].axis('off')
                
                displayed_samples += 1
            
            if displayed_samples >= num_samples:
                break
    
    plt.tight_layout()
    plt.savefig("prediction_comparison.png", bbox_inches='tight')
    plt.show()
    print("Hình ảnh đã được lưu tại: prediction_comparison.png")