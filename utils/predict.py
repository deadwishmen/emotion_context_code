import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import Subset, DataLoader
def predict_and_show(models, device, data_loader, sentences, num_samples=10, class_names=None, conbine=False, thresholds_path='./thresholds.npy'):
    """
    Dự đoán nhãn cho một số mẫu ngẫu nhiên từ data_loader và hiển thị ảnh với nhãn dự đoán, nhãn thực, và đoạn văn bản.

    Args:
        models: Tuple chứa (model_context, model_body, model_face, model_text, fusion_model)
        device: Thiết bị để chạy mô hình (CPU hoặc GPU)
        data_loader: DataLoader chứa dữ liệu kiểm tra
        sentences: Danh sách các đoạn văn bản tương ứng với các mẫu trong dataset
        num_samples: Số lượng mẫu cần hiển thị (default: 10)
        class_names: Danh sách tên các lớp (26 lớp cảm xúc). Nếu None, sử dụng chỉ số lớp.
        conbine: Nếu là 'q_former', xử lý đặc biệt cho pred_context và pred_text
        thresholds_path: Path to the precomputed thresholds file (default: './thresholds.npy').
                        Must be a NumPy array of shape [26] containing per-class thresholds for logits.
    """
    # Validate num_samples
    if num_samples != 10:
        print(f"Warning: num_samples set to 10 for random sampling, ignoring input {num_samples}")
        num_samples = 10
    
    # Load thresholds
    try:
        thresholds = np.load(thresholds_path)
        print(f"Loaded thresholds from '{thresholds_path}'")
    except FileNotFoundError:
        raise FileNotFoundError(f"Thresholds file '{thresholds_path}' not found")
    
    # Validate thresholds
    if not isinstance(thresholds, np.ndarray) or thresholds.shape != (26,):
        raise ValueError("thresholds must be a NumPy array of shape (26,)")
    
    # Convert thresholds to torch tensor
    thresholds = torch.from_numpy(thresholds).to(device)
    
    print(f"num_samples: {num_samples}, device: {device}, conbine: {conbine}")
    print(f"DataLoader length: {len(data_loader)}")
    print(f"Thresholds: {thresholds.cpu().numpy().tolist()}")

    model_context, model_body, model_face, model_text, fusion_model = models
    
    # Chuyển các mô hình sang device và đặt ở chế độ đánh giá
    for model in models:
        model.to(device).eval()
    
    # Nếu không có class_names, sử dụng chỉ số lớp
    if class_names is None:
        class_names = [f"Class {i}" for i in range(26)]
    
    # Lấy số lượng mẫu tổng cộng từ dataset
    dataset_size = len(data_loader.dataset)
    if dataset_size < num_samples:
        raise ValueError(f"Dataset has only {dataset_size} samples, but {num_samples} are required")
    
    # Kiểm tra độ dài của sentences
    if len(sentences) != dataset_size:
        raise ValueError(f"Length of sentences ({len(sentences)}) does not match dataset size ({dataset_size})")
    
    # Chọn ngẫu nhiên 10 mẫu
    random_indices = random.sample(range(dataset_size), num_samples)
    print(f"Selected random indices: {random_indices}")
    
    # Tạo Subset và DataLoader mới cho các mẫu ngẫu nhiên
    random_subset = Subset(data_loader.dataset, random_indices)
    random_loader = DataLoader(
        random_subset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        collate_fn=data_loader.collate_fn if data_loader.collate_fn else None
    )
    
    # Tạo figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples), dpi=80)
    if num_samples == 1:
        axes = [axes]
    
    displayed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(random_loader, total=len(random_loader))):
            try:
                # Unpack batch
                images_context, images_body, images_face, tokenizer_text, labels_cat, labels_cont = batch
                print(f"Batch {batch_idx}: images_context shape: {images_context.shape}, labels_cat shape: {labels_cat.shape}")
                
                # Chuyển dữ liệu sang device
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                images_face = images_face.to(device)
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
                tokenizer_text = {key: val.to(device) for key, val in tokenizer_text.items()}
                labels_cat = labels_cat.to(device)
                
                # Dự đoán
                print("Running model inferences...")
                pred_body = model_body(images_body)
                pred_face = model_face(images_face)
                if conbine == "q_former":
                    pred_text = model_text(**tokenizer_text).last_hidden_state
                    pred_context = model_context(images_context)[:, 1:]
                else:
                    pred_text = model_text(**tokenizer_text).last_hidden_state.mean(dim=1)
                    pred_context = model_context(images_context)
                pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)
                
                # So sánh logits với ngưỡng
                bool_cat_pred = torch.gt(pred_cat, thresholds).cpu().numpy()
                pred_cat_values = pred_cat.cpu().numpy()
                labels_cat = labels_cat.cpu().numpy()
                
                # Xử lý từng mẫu trong batch
                for i in range(images_context.shape[0]):
                    if displayed_samples >= num_samples:
                        break
                    
                    # Lấy chỉ số gốc của mẫu từ random_indices
                    sample_idx = random_indices[displayed_samples]
                    
                    # Lấy ảnh context
                    img = images_context[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    
                    # Lấy nhãn dự đoán từ bool_cat_pred
                    pred_labels = []
                    for idx in range(len(bool_cat_pred[i])):
                        if bool_cat_pred[i][idx]:
                            pred_labels.append((class_names[idx], pred_cat_values[i][idx]))
                    
                    # Sắp xếp theo giá trị logits giảm dần
                    pred_labels.sort(key=lambda x: x[1], reverse=True)
                    pred_label_names = [f"{name} ({prob:.2f})" for name, prob in pred_labels]
                    
                    # Lấy nhãn thực
                    true_labels = np.where(labels_cat[i] == 1)[0]
                    true_label_names = [class_names[idx] for idx in true_labels]
                    
                    # Lấy đoạn văn bản tương ứng
                    sentence = sentences[sample_idx]
                    print(f"Sentence: {sentence}")
                    # Hiển thị ảnh và thông tin
                    axes[displayed_samples].imshow(img)
                    axes[displayed_samples].set_title(
                        f"Pred: {', '.join(pred_label_names) if pred_label_names else 'None'}\n"
                        f"True: {', '.join(true_label_names) if true_label_names else 'None'}\n"
                        f"Text: {sentence}",
                        fontsize=10
                    )
                    axes[displayed_samples].axis('off')
                    
                    displayed_samples += 1
                    print(f"Displayed sample {displayed_samples}/{num_samples}")
                
                if displayed_samples >= num_samples:
                    break
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    if displayed_samples == 0:
        print("Warning: No samples were displayed. Check DataLoader or model inputs.")
        plt.close(fig)
        return
    
    plt.tight_layout()
    plt.savefig("prediction_comparison.png", bbox_inches='tight')
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {str(e)}")
    print("Hình ảnh đã được lưu tại: prediction_comparison.png")