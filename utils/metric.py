from typing import List
import torch
import statistics
import torch.nn.functional as F
from tqdm import tqdm


class MetricStorage:
    """
    A class to store metric values (specifically IoU and Dice scores)
    and compute their overall mean.
    """

    def __init__(self):
        """Initializes empty lists to store the scores."""
        self.dog_iou_scores: List[float] = []
        self.cat_iou_scores: List[float] = []
        self.bg_iou_scores: List[float] = []

        self.dog_dice_scores: List[float] = []
        self.cat_dice_scores: List[float] = []
        self.bg_dice_scores: List[float] = []

        self.dog_accuracy_scores: List[float] = []
        self.cat_accuracy_scores: List[float] = []
        self.bg_accuracy_scores: List[float] = []

        # for point clip
        self.obj_iou_scores: List[float] = []
        self.obj_dice_scores: List[float] = []
        self.obj_accuracy_scores: List[float] = []

        self.val_loss: List[float] = []

    def print_test_scores(self):
        mean_dog_iou_score = statistics.mean(self.dog_iou_scores)
        mean_cat_iou_score = statistics.mean(self.cat_iou_scores)
        mean_bg_iou_score = statistics.mean(self.bg_iou_scores)

        mean_dog_dice_score = statistics.mean(self.dog_dice_scores)
        mean_cat_dice_score = statistics.mean(self.cat_dice_scores)
        mean_bg_dice_score = statistics.mean(self.bg_dice_scores)

        mean_dog_accuracy_score = statistics.mean(self.dog_accuracy_scores)
        mean_cat_accuracy_score = statistics.mean(self.cat_accuracy_scores)
        mean_bg_accuracy_score = statistics.mean(self.bg_accuracy_scores)

        mean_iou_score = statistics.mean(
            [mean_dog_iou_score, mean_cat_iou_score, mean_bg_iou_score]
        )
        mean_dice_score = statistics.mean(
            [mean_dog_dice_score, mean_cat_dice_score, mean_bg_dice_score]
        )
        mean_accuracy_score = statistics.mean(
            [mean_dog_accuracy_score, mean_cat_accuracy_score, mean_bg_accuracy_score]
        )
        print(
            f"{mean_dog_iou_score=:.02f}\n"
            f"{mean_cat_iou_score=:.02f}\n"
            f"{mean_bg_iou_score=:.02f}\n"
            f"{mean_dog_dice_score=:.02f}\n"
            f"{mean_cat_dice_score=:.02f}\n"
            f"{mean_bg_iou_score=:.02f}\n"
            f"{mean_dog_accuracy_score=:.02f}\n"
            f"{mean_cat_accuracy_score=:.02f}\n"
            f"{mean_bg_accuracy_score=:.02f}\n"
            f"{mean_iou_score=:.02f}\n"
            f"{mean_dice_score=:.02f}\n"
            f"{mean_accuracy_score=:.02f}\n"
        )

    def get_mean_dice_score(self):
        mean_dog_dice_score = statistics.mean(self.dog_dice_scores)
        mean_cat_dice_score = statistics.mean(self.cat_dice_scores)
        mean_bg_dice_score = statistics.mean(self.bg_dice_scores)

        return statistics.mean(
            [mean_dog_dice_score, mean_cat_dice_score, mean_bg_dice_score]
        )

    def print_test_scores_pointclip(self):
        mean_bg_iou_score = statistics.mean(self.bg_iou_scores)
        mean_bg_dice_score = statistics.mean(self.bg_dice_scores)
        mean_bg_accuracy_score = statistics.mean(self.bg_accuracy_scores)

        mean_obj_iou_score = statistics.mean(self.obj_iou_scores)
        mean_obj_dice_score = statistics.mean(self.obj_dice_scores)
        mean_obj_accuracy_score = statistics.mean(self.obj_accuracy_scores)

        mean_iou_score = statistics.mean([mean_obj_iou_score, mean_bg_iou_score])
        mean_dice_score = statistics.mean(
            [mean_obj_dice_score, mean_bg_dice_score]
        )
        mean_accuracy_score = statistics.mean([mean_obj_accuracy_score, mean_bg_accuracy_score])
        
        print(
            f"{mean_bg_iou_score=:.02f}\n"
            f"{mean_obj_iou_score=:.02f}\n"
            
            f"{mean_bg_dice_score=:.02f}\n"
            f"{mean_obj_dice_score=:.02f}\n"

            f"{mean_bg_accuracy_score=:.02f}\n"
            f"{mean_obj_accuracy_score=:.02f}\n"
            
            f"{mean_iou_score=:.02f}\n"
            f"{mean_dice_score=:.02f}\n"
            f"{mean_accuracy_score=:.02f}\n"
        )

def compute_iou(preds, targets, eps=1e-8):
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = intersection / (union + eps)
    return iou.item()


def compute_accuracy(preds, targets):
    num_correct = (preds == targets).sum()
    num_pixels = torch.numel(preds)

    return num_correct / num_pixels


def compute_dice_coefficient(preds, targets, eps=1e-8):
    intersection = (preds * targets).sum()
    dice = (2 * intersection) / (preds.sum() + targets.sum() + eps)
    return dice.item()


def check_accuracy(loader, model, metric: MetricStorage, loss_fn=None, device="cuda"):
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for image, mask in loop:
            image = image.float().to(device)
            # print(f"input shape: {x.size()}") # input shape: torch.Size([64, 3, 256, 256])

            mask = mask.long().to(device)
            # print(f"target mask shape: {y.size()}")  # target mask shape: torch.Size([64, 1, 256, 256])

            output = model(image)

            if loss_fn != None:
                loss = loss_fn(output, mask)
                metric.val_loss.append(loss.item())
            # print(f"Evaluation Loss: {loss}")
            # print(f"output shape: {output.size()}") # output shape: torch.Size([64, 3, 256, 256])
            probabilities = F.softmax(output, dim=1)
            # print(f"probs shape: {probabilities.size()}") # probs shape: torch.Size([64, 3, 256, 256])

            pred_classes = torch.argmax(probabilities, dim=1)
            # print(f"pred argmax shape: {pred_classes.size()}") # pred argmax shape: torch.Size([64, 1, 256, 256])

            # background
            pred_bg_mask = (pred_classes == 0).float()
            actual_bg_mask = (mask == 0).float()
            bg_dice_score = compute_dice_coefficient(pred_bg_mask, actual_bg_mask)
            bg_iou_score = compute_iou(pred_bg_mask, actual_bg_mask)
            bg_accuracy_score = compute_accuracy(pred_bg_mask, actual_bg_mask)

            metric.bg_iou_scores.append(bg_iou_score)
            metric.bg_dice_scores.append(bg_dice_score)
            metric.bg_accuracy_scores.append(bg_accuracy_score)

            # cat
            pred_cat_mask = (pred_classes == 1).float()
            actual_cat_mask = (mask == 1).float()
            cat_dice_score = compute_dice_coefficient(pred_cat_mask, actual_cat_mask)
            cat_iou_score = compute_iou(pred_cat_mask, actual_cat_mask)
            cat_accuracy_score = compute_accuracy(pred_cat_mask, actual_cat_mask)

            metric.cat_iou_scores.append(cat_iou_score)
            metric.cat_dice_scores.append(cat_dice_score)
            metric.cat_accuracy_scores.append(cat_accuracy_score)

            # dog
            pred_dog_mask = (pred_classes == 2).float()
            actual_dog_mask = (mask == 2).float()
            dog_dice_score = compute_dice_coefficient(pred_dog_mask, actual_dog_mask)
            dog_iou_score = compute_iou(pred_dog_mask, actual_dog_mask)
            dog_accuracy_score = compute_accuracy(pred_dog_mask, actual_dog_mask)

            metric.dog_iou_scores.append(dog_iou_score)
            metric.dog_dice_scores.append(dog_dice_score)
            metric.dog_accuracy_scores.append(dog_accuracy_score)

            loop.set_postfix()
    model.train()


def check_accuracy_pointclip(loader, model, metric: MetricStorage, device="cuda"):
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, point, mask) in enumerate(loop):
            # print(f"batch: {batch_idx} images: {images.size()}, points: {points.size()}, masks: {masks.size()}")
            image = image.float().to(device=device)
            mask = mask.float().to(device=device)  # batch, class, height, width
            point = point.float().to(device=device)
            output_logits = model(image, point)

            # Threshold logits directly at 0
            predictions_bool = output_logits > 0

            # Convert boolean predictions to integers (0 or 1)
            predictions_int = predictions_bool.int()

            # background
            pred_bg_mask = (predictions_int == 0).float()
            actual_bg_mask = (mask == 0).float()
            bg_dice_score = compute_dice_coefficient(pred_bg_mask, actual_bg_mask)
            bg_iou_score = compute_iou(pred_bg_mask, actual_bg_mask)
            bg_accuracy_score = compute_accuracy(pred_bg_mask, actual_bg_mask)

            metric.bg_iou_scores.append(bg_iou_score)
            metric.bg_dice_scores.append(bg_dice_score)
            metric.bg_accuracy_scores.append(bg_accuracy_score)
            
            # object
            pred_obj_mask = (predictions_int == 1).float()
            actual_obj_mask = (mask == 1).float()
            obj_dice_score = compute_dice_coefficient(pred_obj_mask, actual_obj_mask)
            obj_iou_score = compute_iou(pred_obj_mask, actual_obj_mask)
            obj_accuracy_score = compute_accuracy(pred_obj_mask, actual_obj_mask)

            metric.obj_iou_scores.append(obj_iou_score)
            metric.obj_dice_scores.append(obj_dice_score)
            metric.obj_accuracy_scores.append(obj_accuracy_score)
            
            loop.set_postfix()
    model.train()
