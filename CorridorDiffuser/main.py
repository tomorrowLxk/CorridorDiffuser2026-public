import os
import torch
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, DDIMScheduler
from dataclasses import dataclass

from model import SafetyCorridorUNet
from dataloader import SafetyCorridorDataset, generate_dataloader
from losses import safe_loss
from eval import evaluate
from tqdm import tqdm


@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 4
    test_batch_size = 1
    num_epochs = 200
    gradient_accumulation_steps = 32
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"
    seed = 42
    VAL_RATIO = 0.13
    TRAIN_RATIO = 0.8
    output_dir = "output1"
    MAP_DIR = r"data/maps"
    TARGET_DIR = r"data/targets"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    noise_T = 1000


def train_loop(config, model, train_scheduler, eval_scheduler, optimizer, dataset):
    model = model.to(config.DEVICE)
    # model.load_state_dict(torch.load(fr"checkpoints/best.pth",
    #                                  map_location=config.DEVICE))
    train_dataloader, val_dataloader, test_dataloader = generate_dataloader(dataset, config)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
    weights_dir = os.path.join(config.output_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, "logs"))

    use_amp = config.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for step, (batch) in enumerate(progress_bar):
            # 读取数据
            clean_images = batch["Y_gt"].to(config.DEVICE)
            map_tensor = batch["map"].to(config.DEVICE)

            # 添加噪声
            noise = torch.randn_like(clean_images).to(config.DEVICE)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, train_scheduler.config.num_train_timesteps, (bs,), device=config.DEVICE).long()
            noisy_images = train_scheduler.add_noise(clean_images, noise, timesteps)

            # 梯度清零
            if step % config.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # 前向 + 损失 + 反向
            with torch.amp.autocast('cuda', enabled=use_amp):
                noise_pred = model(noisy_images, timesteps, map_tensor)
                loss_dic = safe_loss(noise, noise_pred, clean_images, noisy_images, map_tensor, timesteps,
                                     train_scheduler)
                # total_loss = (loss_dic["noise_loss"] +
                #               0.01 * loss_dic['dice_loss'] +
                #               0.2 * loss_dic['sg_loss'] +
                #               0.1 * loss_dic['obstacles_loss']) / config.gradient_accumulation_steps
                # total_loss = (loss_dic["noise_loss"] + 0.2 * loss_dic['images_loss'])
                if epoch < -1:
                    total_loss = (loss_dic["noise_loss"] +
                                  0.1 * loss_dic['sg_loss'] +
                                  0.1 * loss_dic['obstacles_loss']) / config.gradient_accumulation_steps
                else:
                    total_loss = (loss_dic["noise_loss"] +
                                  0.2 * loss_dic['dice_loss'] +
                                  0.2 * loss_dic['images_loss'] +
                                  0.1 * loss_dic['sg_loss'] +
                                  0.1 * loss_dic['obstacles_loss']) / config.gradient_accumulation_steps

            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # 梯度累积步数到了，才更新
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()

            # 更新进度条，更新日志
            logs = {
                "loss": total_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)

            writer.add_scalar("train/noise_loss", loss_dic["noise_loss"].detach().item(), global_step)
            writer.add_scalar("train/images_loss", loss_dic["images_loss"].detach().item(), global_step)
            writer.add_scalar("train/dice_loss", loss_dic["dice_loss"].detach().item(), global_step)
            writer.add_scalar("train/sg_loss", loss_dic["sg_loss"].detach().item(), global_step)
            writer.add_scalar("train/obstacles_loss", loss_dic["obstacles_loss"].detach().item(), global_step)
            writer.add_scalar("train/lr", logs["lr"], global_step)

            global_step += 1

        # 执行完了再来最后的更新
        if (step + 1) % config.gradient_accumulation_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            model.eval()
            evaluate(model, val_dataloader, eval_scheduler, epoch, config)
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            save_path = os.path.join(weights_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
        # if epoch + 1 == config.num_epochs:
        #     evaluate(model, test_dataloader, eval_scheduler, epoch, config, iftest=True)

    writer.close()


if __name__ == '__main__':
    config = TrainingConfig()
    dataset = SafetyCorridorDataset(map_dir=config.MAP_DIR, target_dir=config.TARGET_DIR)
    model = SafetyCorridorUNet(encoder_type='vit')

    train_scheduler = DDPMScheduler(num_train_timesteps=config.noise_T)
    eval_scheduler = DDPMScheduler(num_train_timesteps=config.noise_T)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    train_loop(config, model, train_scheduler, eval_scheduler, optimizer, dataset)

