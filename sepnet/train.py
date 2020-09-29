import torch
import torchvision
from torch.utils.data import DataLoader
from sepnet.utils import mix_images, unstack_images
from sepnet.loss import SeperationLoss
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, config, trn_dataset, tst_dataset):
        self.config = config
        self.model = model
        self.trn_dataset = trn_dataset
        self.tst_dataset = tst_dataset
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def train(self):
        config, device = self.config, self.device
        model = self.model.to(device)
        criterion = SeperationLoss(loss=torch.nn.SmoothL1Loss())
        optimizer = torch.optim.SGD(
            model.parameters(), **config.opimizer_params
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )
        trn_loader = DataLoader(self.trn_dataset, shuffle=True,
                                pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
        tst_loader = DataLoader(self.tst_dataset, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

        pbar = tqdm(range(config.epochs), total=config.epochs)

        for epoch in pbar:

            def run_epoch(is_train, loader, return_samples=False):
                losses = []
                for images in loader:
                    if isinstance(images, tuple):
                        images = images[0]
                    images = images.to(device)
                    image_pairs, mixed_images, _ = mix_images(images)
                    constructed_images = model(mixed_images)
                    loss = criterion(constructed_images, image_pairs)
                    losses.append(loss.item())
                    if is_train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_norm_clip
                        )
                        optimizer.step()
                        scheduler.step()
                if return_samples:
                    return losses, constructed_images
                return losses

            @torch.no_grad()
            def save_progress(constructed_images):
                images = unstack_images(constructed_images)
                image_out_dir = 'img_outputs'
                if not os.path.exists(image_out_dir):
                    os.mkdir(image_out_dir)
                torchvision.utils.save_image(
                    images,
                    fp=os.path.join(
                        image_out_dir,
                        f'output_epoch{epoch + 1}.jpg'
                    )
                )
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        config.model_path,
                        f'SepNet_Epoch{epoch + 1}.pt'
                    )
                )

            model.train()
            train_epoch_loss = run_epoch(is_train=True, loader=trn_loader)
            mean_train_epoch_loss = sum(
                train_epoch_loss) / len(train_epoch_loss)

            model.eval()
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    test_epoch_loss, constructed_images = run_epoch(
                        is_train=False,
                        loader=tst_loader,
                        return_samples=True
                    )
                save_progress(constructed_images)
            else:
                with torch.no_grad():
                    test_epoch_loss = run_epoch(
                        is_train=False, loader=tst_loader
                    )
            mean_test_epoch_loss = sum(test_epoch_loss) / len(test_epoch_loss)
            tqdm.write(
                f'Epoch {epoch + 1}/{config.epochs}, \
                    Train loss: {mean_train_epoch_loss:.3f}, \
                        Test loss: {mean_test_epoch_loss:.3f}'
            )
