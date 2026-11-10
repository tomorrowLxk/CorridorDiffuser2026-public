import os
import torch
import numpy as np
from matplotlib import pyplot as plt


@torch.no_grad()
def evaluate(model, val_dataloader, eval_scheduler, epoch, config, iftest = False):
    # 获取一个 batch 的数据
    # 创建保存目录
    eval_dir = os.path.join(config.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    epoch_dir = os.path.join(eval_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    test_dir = os.path.join(config.output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    process_dir = os.path.join(epoch_dir, "process")
    os.makedirs(process_dir, exist_ok=True)

    if iftest:
        for batch in val_dataloader:
            idx = int(batch['idx'])
            map_tensor = batch["map"][:1].to(config.DEVICE)
            clean_images = batch["Y_gt"][:1].to(config.DEVICE)
            noisy_images = torch.randn_like(clean_images).to(config.DEVICE)
            bs = map_tensor.shape[0]
            # 画出地图
            plt.imshow(np.transpose(np.array(map_tensor[0].cpu()+ 1) / 2, (1, 2, 0)), origin='lower')
            plt.axis('off')
            plt.savefig(os.path.join(test_dir, f"{idx}_map.png"), bbox_inches='tight', dpi=600)
            plt.close()
            # 画出走廊
            with torch.no_grad():
                for t in eval_scheduler.timesteps:
                    timestamps = torch.full((bs,), t, dtype=torch.long).to(config.DEVICE)
                    noise_pred = model(noisy_images, timestamps, map_tensor)
                    noisy_images = eval_scheduler.step(noise_pred, t, noisy_images).prev_sample
            pred_images = noisy_images.clip(-1, 1)
            npy_pred_image = (pred_images.detach().cpu().numpy() + 1) / 2
            plt.imshow(npy_pred_image[0,0], origin='lower')
            plt.axis('off')
            plt.savefig(os.path.join(test_dir, f"{idx}_corridor.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()
            # 保存结果
            np.save(os.path.join(test_dir, f"{idx}_VASC.npy"), npy_pred_image[0])
    else:
        for batch in val_dataloader:
            map_tensor = batch["map"][:1].to(config.DEVICE)
            clean_images = batch["Y_gt"][:1].to(config.DEVICE)

            bs = map_tensor.shape[0]

            # 保存无噪声图片
            npy_clean_image = (clean_images.detach().cpu().numpy() + 1) / 2
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            axs = visualize_float_result(npy_clean_image[:, 0], axs)
            plt.savefig(os.path.join(epoch_dir, f"clean_images.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # 初始化噪声
            noisy_images = torch.randn_like(clean_images).to(config.DEVICE)
            # 反向扩散过程
            with torch.no_grad():
                for t in eval_scheduler.timesteps:
                    # 将噪声调度到当前时间步长
                    timestamps = torch.full((bs,), t, dtype=torch.long).to(config.DEVICE)
                    noise_pred = model(noisy_images, timestamps, map_tensor)
                    noisy_images = eval_scheduler.step(noise_pred, t, noisy_images).prev_sample

                    if (t+1) % 50 == 0 or t==0:
                        npy_noisy_image = (noisy_images.clip(-1, 1).detach().cpu().numpy() + 1) / 2
                        # fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                        # axs = visualize_float_result(npy_noisy_image[:, 0], axs)
                        plt.imshow(npy_noisy_image[0,0], origin='lower')
                        plt.axis('off')
                        plt.savefig(os.path.join(process_dir,f"noisy_images_{t}.png"), bbox_inches='tight', dpi=600)
                        plt.close()

            pred_images = noisy_images.clip(-1, 1)  # 最终得到的去噪后的图像

            # 保存生成的图像和真实图像
            npy_pred_image = (pred_images.detach().cpu().numpy() + 1) / 2
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            axs = visualize_float_result(npy_pred_image[:, 0], axs)
            plt.savefig(os.path.join(epoch_dir, f"pred_images.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()
            break


def visualize_float_result(image, axs):
    for i, img in enumerate(image):
        im = axs[i // 2, i % 2].imshow(img)
        plt.colorbar(im, ax=axs[i // 2, i % 2])
        axs[i // 2, i % 2].axis('off')
    return axs
