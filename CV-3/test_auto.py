import torch
import torch.nn.functional as F
from tqdm import tqdm
import utils
from autoencoder import AutoencoderWithSegmentationHead
from autoencoder import Autoencoder
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder().to(device)
from utils import MetricStorage




def test():
    print("Testing...")

    # 加载测试数据
    test_loader = utils.get_test_loader(batch_size=64, num_workers=4, pin_memory=True)

    if test_loader is None:
        raise RuntimeError("Error: Failed to load test data.")

    # 加载训练好的模型
    model_checkpoint_path = "segmentation_model_checkpoint_48.pth.tar" 
    model = AutoencoderWithSegmentationHead(autoencoder.encoder, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device)["state_dict"], strict=False)
    model.eval()

    metric = utils.MetricStorage()
    # 调用check_accuracy计算IoU、Dice和Accuracy
    utils.check_accuracy(test_loader, model, metric, device=device, filename="segmentation_test_results")

    metric.print_latest_scores()

if __name__ == "__main__":
    test()