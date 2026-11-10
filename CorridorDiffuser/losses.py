import torch
import torch.nn.functional as F


def safe_loss(noise, noise_pred, clean_images, noisy_images, map_tensor, timesteps, scheduler):
    B, C, H, W = clean_images.shape

    obstacles_imgs = map_tensor[:, 2].unsqueeze(1)  # (B, 1, H, W)
    sg_imgs = map_tensor[:, 0:2]  # (B, 2, H, W)

    obstacles_mask = (obstacles_imgs > 0).float()  # 1=障碍物, 0=非障碍物
    sg_mask = (sg_imgs > 0).float().sum(1, keepdim=True).clamp(max=1.0)  # 合并起终点，0/1 掩码

    # 0. 预测去噪图像
    alphas_cumprod = scheduler.alphas_cumprod.to(noisy_images.device)
    alpha_t = alphas_cumprod[timesteps].view(B, 1, 1, 1)  # (B, 1, 1, 1) 便于广播
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    pred_images = (noisy_images - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    pred_images = torch.clamp(pred_images, -1.0, 1.0)

    # 1. 计算噪声误差
    noise_loss = F.mse_loss(noise_pred, noise)

    # 2. 计算图像误差
    images_loss = F.mse_loss(pred_images, clean_images)

    # 3. Dice Loss
    pred_prob = torch.sigmoid((pred_images + 0.975) * 200)  # 映射到0~1
    target_prob = (clean_images > -1).float()  # 0/1

    inter = (pred_prob * target_prob).sum(dim=(1, 2, 3))
    union = pred_prob.sum(dim=(1, 2, 3)) + target_prob.sum(dim=(1, 2, 3))
    dice_loss = 1 - (2 * inter + 1e-6) / (union + 1e-6)
    dice_loss = dice_loss.mean()

    # 4. 计算起终点损失，起终点附近要准确
    sg_loss = F.mse_loss(pred_images * sg_mask, clean_images * sg_mask)

    # 5. 计算障碍物损失，障碍物处pred_images应该为0
    obstacles_loss = F.mse_loss(pred_images * obstacles_mask, -obstacles_mask)

    loss_dict = {
        "noise_loss": noise_loss,
        "images_loss": images_loss,
        "dice_loss": dice_loss,
        "sg_loss": sg_loss,
        "obstacles_loss": obstacles_loss
    }

    return loss_dict

