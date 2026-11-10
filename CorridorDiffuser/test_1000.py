# 使用模型生成最终结果的图像
import os
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from dataclasses import dataclass
import numpy as np
from model import SafetyCorridorUNet
from dataloader import SafetyCorridorDataset, generate_dataloader
from tqdm import tqdm
import cv2
import time

TIME_COST = []


@dataclass
class TestConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 4
    test_batch_size = 1
    mixed_precision = "fp16"
    seed = 42
    VAL_RATIO = 0.19
    TRAIN_RATIO = 0.8
    output_dir = "result_1000"
    MAP_DIR = r"data/maps"
    TARGET_DIR = r"data/targets"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    noise_T = 1000


def test_draw(config, model, test_scheduler, dataset):
    torch.backends.cudnn.benchmark = True
    model = model.to(config.DEVICE)
    # model.load_state_dict(torch.load(fr"checkpoints/best.pth",
    #                                  map_location=config.DEVICE))
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()
    _, _, test_dataloader = generate_dataloader(dataset, config)

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    pic_dir = os.path.join(config.output_dir, 'pic')
    os.makedirs(pic_dir, exist_ok=True)

    for batch in tqdm(test_dataloader):
        idx = batch['idx']
        map_tensor = batch["map"].to(config.DEVICE)
        clean_images = batch["Y_gt"].to(config.DEVICE)
        noisy_images = torch.randn_like(clean_images).to(config.DEVICE)
        bs = map_tensor.shape[0]
        # 先计算走廊
        t1 = time.time()
        with torch.no_grad():
            for t in test_scheduler.timesteps:
                timestamps = torch.full((bs,), t, dtype=torch.long).to(config.DEVICE)
                noise_pred = model(noisy_images, timestamps, map_tensor)
                noisy_images = test_scheduler.step(noise_pred, t, noisy_images).prev_sample
        pred_images = noisy_images.clip(-1, 1)
        TIME_COST.append(time.time() - t1)
        npy_pred_image = (pred_images.detach().cpu().numpy() + 1) / 2

        # for i in range(bs):
        #     id_i = int(idx[i])
        #     roadmap = (map_tensor[i].detach().cpu().numpy() + 1) / 2
        #     obs = roadmap[2]
        #     sg = roadmap[0] + roadmap[1]
        #     corridor = npy_pred_image[i, 0]
        #     rgb_image = np.dstack((obs, sg, corridor))
        #     cv2.imwrite(os.path.join(pic_dir, f"{id_i}_ALL.png"), (rgb_image * 255).astype(np.uint8))
        #     cv2.imwrite(os.path.join(pic_dir, f"{id_i}_corridor.png"),
        #                 np.expand_dims(corridor * 255, axis=-1).astype(np.uint8))
        #     np.save(os.path.join(pic_dir, f"{id_i}_VASC.npy"), corridor)


if __name__ == '__main__':
    config = TestConfig()
    dataset = SafetyCorridorDataset(map_dir=config.MAP_DIR, target_dir=config.TARGET_DIR)
    model = SafetyCorridorUNet(encoder_type='iia')
    test_scheduler = DDPMScheduler(num_train_timesteps=config.noise_T)
    # test_scheduler.set_timesteps(num_inference_steps=1000)
    test_draw(config, model, test_scheduler, dataset)
    print(np.mean(TIME_COST))
    np.save(os.path.join(config.output_dir, "TIME_COST.npy"), TIME_COST)

