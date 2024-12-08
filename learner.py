import numpy as np
import os
import random
import logging
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import DiffAR
from eval import run_metrics
import distrib as distrib
from utils_for_inference import LogMelSpectrogram

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # DEBUG


class DiffARLearner:
    def __init__(self, model, train_ds, valid_ds, test_ds, is_master, params, *args, **kwargs):
        self.model_dir = params.model_dir
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.optimizer = torch.optim.Adam(self.model.parameters(), params.learning_rate)
        self.params = params
        self.step = 0
        self.is_master = is_master
        self.device = next(self.model.parameters()).device
        if params.spec_loss_coeff > 0.:
            self.log_mel_spec = LogMelSpectrogram(n_mels = 80).to(self.device)

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.feature_extractor = instantiate(params.features)

        # data augmentations
        augment = []
        if len(params.augment) > 0:
            for augmentation in params.augment:
                augment.append(instantiate(params[augmentation]))
        self.augment = nn.Sequential(*augment)

        self.restore_from_checkpoint()
        if is_master and not params.test:
            self.summary_writer = SummaryWriter(os.getcwd(), purge_step=self.step)

        logger.info(f"running in: {os.getcwd()}")
        logger.info(f"model size: {sum(p.numel() for p in model.parameters()):,}")

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
                'step': self.step,
                'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
                'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
                'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{os.getcwd()}/{save_basename}'
        link_name = f'{os.getcwd()}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)
        # maintain only last 3 checkpoints (+1 which is the symlink)
        path = os.getcwd()
        path = path.replace("[", "\[").replace("]", "\]")
        os.system(f"rm `ls -t {path}/*.pt | awk 'NR>4'`")

    def save_to_min_checkpoint(self, filename='weights', tgt_folder = 'min'):
        save_basename = f'{filename}-{self.step}.pt'
        os.system(f"mkdir -p {os.getcwd()}/{tgt_folder}")
        save_name = f'{os.getcwd()}/{tgt_folder}/{save_basename}'
        link_name = f'{os.getcwd()}/{tgt_folder}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)
        # maintain only last 3 checkpoints (+1 which is the symlink)
        path = f'{os.getcwd()}/{tgt_folder}/'
        # path = os.getcwd()
        path = path.replace("[", "\[").replace("]", "\]")
        os.system(f"rm `ls -t {path}/*.pt | awk 'NR>4'`")

    def restore_from_checkpoint(self, filename='weights'):
        try:
            path = f'{os.getcwd()}/{filename}.pt'
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint)
            logger.info(f"loaded checkpoint from: {path} ({checkpoint['step']} steps)")
            return True
        except FileNotFoundError:
            logger.info(f"training from scratch")
            return False

    def train_epoch(self, dataset):
        self.model.train()
        epoch_loss = 0
        epoch_loss_denoise = 0
        epoch_loss_spec = 0
        for data in tqdm(dataset, desc=f'Step: {self.step}') if self.is_master else dataset:
            clean, conditioned_audio, conditoned_phonemes, conditioned_energy, overlap  = data[0].squeeze(1), data[1].squeeze(1), data[2], data[3], data[4]
            #--- value is non-intuitive: 
            #   -1 : don't mask_loss_using_overlap
            #   0  : mask all overlap
            #   n>0: mask all minus n samples
            if self.params.mask_loss_using_overlap >= 0:
                overlap = overlap.to(self.device)
            else:
                overlap = None
            loss, loss_denoise, loss_spec = self.train_step(clean.to(self.device), conditioned_audio.to(self.device), conditoned_phonemes.to(self.device), conditioned_energy.to(self.device), overlap)
            epoch_loss += loss.item()
            if loss_denoise is not None and loss_spec is not None:
                epoch_loss_denoise += loss_denoise.item()
                epoch_loss_spec += loss_spec.item()
            if torch.isnan(loss).any():
                raise RuntimeError(f'Detected NaN loss at step {self.step}.')
            self.step += 1
        epoch_loss /= len(self.train_ds)
        epoch_loss = distrib.average([epoch_loss])[0]
        
        if epoch_loss_denoise > 0 and epoch_loss_spec > 0:
            epoch_loss_denoise /= len(self.train_ds)
            epoch_loss_denoise = distrib.average([epoch_loss_denoise])[0]
            
            epoch_loss_spec /= len(self.train_ds)
            epoch_loss_spec = distrib.average([epoch_loss_spec])[0]
            logger.info(f'train denoise loss: {epoch_loss_denoise:.4f} spec loss: {epoch_loss_spec:.4f}')
        
        return epoch_loss

    def eval_epoch(self, dataset):
        ### There is an option calculating pseq, stoi ###
        self.model.eval()
        # all_pesq, all_stoi, n = 0, 0, 0
        n = 0
        epoch_loss_val = 0
        for data in tqdm(dataset, desc=f'evaluating') if self.is_master else dataset:
            clean, conditioned_audio, conditoned_phonemes, conditioned_energy, overlap  = data[0].squeeze(1), data[1].squeeze(1), data[2], data[3], data[4]

            if self.params.mask_loss_using_overlap >= 0:
                overlap = overlap.to(self.device)
            else:
                overlap = None
            # pred_audio = self.valid_step(noisy.to(self.device), clean.shape[1]).cpu(), text_grid_clean.to(self.device)
            # pesq_sc, stoi_sc = run_metrics(clean, pred_audio, self.params.sample_rate)
            # all_pesq += pesq_sc
            # all_stoi += stoi_sc
            # n += clean.shape[0]
            # noisy1 = (noisy - clean)
            # noisy = noisy1 + self.augment(clean)

            loss_val, loss_denoise_val, loss_spec_val = self.valid_loss(clean.to(self.device), conditioned_audio.to(self.device), conditoned_phonemes.to(self.device), conditioned_energy.to(self.device), overlap)
            epoch_loss_val += loss_val.item()
            if torch.isnan(torch.tensor(epoch_loss_val)).any():
                 raise RuntimeError(f'Detected NaN loss at step {self.step}.')
            n += 1
        assert(n==len(self.valid_ds))
        epoch_loss_val /= len(self.valid_ds)
        epoch_loss_val = distrib.average([epoch_loss_val])[0]

        # all_pesq, all_stoi = all_pesq / n, all_stoi / n
        # all_pesq, all_stoi = distrib.average([all_pesq, all_stoi])
        # return all_pesq, all_stoi, epoch_loss_val

        return epoch_loss_val

    def test(self):
        if self.test_ds is None:
            logger.warning('test dataset is not set, skipping test')
            return

        pesq, stoi = self.eval_epoch(self.test_ds)
        logger.info(f"test results:")
        logger.info(f"pesq = {pesq}")
        logger.info(f"stoi = {stoi}")

    def train(self, max_steps=None):
        epoch = 0
        min_loss = 100
        min_loss_val = 100
        while True:
            # === TRAIN LOOP ===
            print("start training")
            loss = self.train_epoch(self.train_ds)
            self.log_dict({"train/loss": loss, "train/grad_norm": self.grad_norm})

            # === VALIDATION LOOP ===
            if epoch % self.params.val_every_n_epochs == 0 and epoch != 0 and self.valid_ds is not None:
                print("choose to do val")
                epoch_loss_val = self.eval_epoch(self.valid_ds)
                self.log_dict({"valid/loss": epoch_loss_val})
                if self.is_master and epoch_loss_val < min_loss_val:
                    print("loss_val < min_loss_val")
                    print("choose to do checkpoint to min_val folder")
                    self.save_to_min_checkpoint(tgt_folder = 'min_val')
                    min_loss_val = epoch_loss_val
                # pesq, stoi, epoch_loss_val = self.eval_epoch(self.valid_ds)
                # self.log_dict({"valid/loss": epoch_loss_val, "valid/pesq": pesq, "valid/stoi": stoi})
               
            if self.is_master and epoch % self.params.summery_every_n_epochs == 0:
                # print("choose to do audio summery")
                # self._audio_summary(self.step)
                print("choose to do checkpoint")
                self.save_to_checkpoint()
                if loss < min_loss:
                    print("loss<min_loss")
                    print("choose to do checkpoint")
                    self.save_to_min_checkpoint()
                    min_loss = loss
            print(self.step)
            if max_steps is not None and self.step >= max_steps:
                print("choose to finish")
                print(self.step)
                return

            epoch += 1

    def get_loss_with_overlap_mask(self, noise, predicted, overlap):
        assert overlap is not None
        assert self.params.mask_loss_using_overlap >= 0, "a call to 'get_loss_with_overlap_mask' should not happen if mask_loss_using_overlap == -1"
        loss = F.l1_loss(noise, predicted.squeeze(1), reduction = 'none')
        
        samples_before = self.params.mask_loss_using_overlap #--- samples before the overlap border to include in the loss
        for row, k in enumerate(overlap):
            k = max(0, k - samples_before)
            loss[row, :k].zero_()
        win_len = loss.shape[1] 
        non_overlap = win_len - overlap + samples_before
        loss = loss / non_overlap.unsqueeze(1) / self.params.batch_size_train
        loss = loss.sum()
        return loss

    def train_step(self, audio, audio_conditioner, phonemes_conditioner, Energy_conditioner, overlap = None):
        # (itamark) setting grads to "None" instead of just calling optimizer.zero_grad(), makes sure params are not changed if grad is zero (as 
        # would happen e.g when using weight decay, momentum, etc. 
        # see e.g https://discuss.pytorch.org/t/in-optimizer-zero-grad-set-p-grad-none/31934/3
        for param in self.model.parameters():
            param.grad = None
        N, T = audio.shape
        device = audio.device
        self.noise_level = self.noise_level.to(device)
        t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
        predicted = self.model(noisy_audio, audio_conditioner.unsqueeze(1), t, phonemes_conditioner, Energy_conditioner)
        if overlap is None:
            loss_denoise = F.l1_loss(noise, predicted.squeeze(1))
        else:
            loss_denoise = self.get_loss_with_overlap_mask(noise, predicted, overlap)
        
        if self.params.spec_loss_coeff > 0.:
            loss_spec = F.l1_loss(self.log_mel_spec(noise), self.log_mel_spec(predicted.squeeze(1)))
            #logger.info(f'denoise loss: {loss_denoise:.4f} spec loss: {loss_spec:.4f}')
            loss_coeff = self.params.spec_loss_coeff
            loss = (1 - loss_coeff) * loss_denoise + loss_coeff * loss_spec
        else:
            loss_spec = None
            loss = loss_denoise

        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.optimizer.step()
        
        return loss, loss_denoise, loss_spec

### TODO: if pseq / stoi  are relevant ###
    # def valid_step(self, original_waveform, audio_text_grid):
    
    #     device = original_waveform.device()
    #     with torch.no_grad():
    #         beta = np.array(self.params.noise_schedule)
    #         alpha = 1 - beta
    #         alpha_cum = np.cumprod(alpha)

    #         # audio_a, (start, end) = sample_segment(audio, self.params.window_length, ret_idx=True)
    #         # List_excisting_phonemes = Build_excisting_phonemes(start, end, audio_text_grid)[None,:]

    #         audio = torch.randn_like(original_waveform, device=device)

    #         for n in range(len(alpha) - 1, -1, -1):
    #             c1 = 1 / alpha[n]**0.5
    #             c2 = beta[n] / (1 - alpha_cum[n])**0.5
    #             ## there is no conditiner for now, and the model isnt using this input
    #             painted_window = torch.zeros(original_waveform.to(device).shape, device=device)
    #             # spectrogram = original_waveform.to(device)
    #             audio = c1 * (audio.to(device) - c2 * self.model(audio, painted_window, torch.tensor([n], device=audio.device), audio_text_grid.to(device)).squeeze(1))
    #             if n > 0:
    #                 noise = torch.randn_like(audio)
    #                 sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
    #                 audio += sigma * noise
    #             # audio = torch.clamp(audio, -1.0, 1.0)
                
    #             device = torch.device('cuda')


    #     return audio
        
    # def valid_step(self, conditioner, n_samples):
    #     device = conditioner.device
    #     with torch.no_grad():
    #         beta = np.array(self.params.noise_schedule)
    #         alpha = 1 - beta
    #         alpha_cum = np.cumprod(alpha)
    #         audio = torch.randn(conditioner.shape[0], n_samples, device=device)
    #         noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    #         for n in range(len(alpha) - 1, -1, -1):
    #             c1 = 1 / alpha[n]**0.5
    #             c2 = beta[n] / (1 - alpha_cum[n])**0.5
    #             audio = c1 * (audio - c2 * self.model(audio, conditioner.unsqueeze(1), torch.tensor([n], device=audio.device)).squeeze(1))
    #             if n > 0:
    #                 noise = torch.randn_like(audio)
    #                 sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
    #                 audio += sigma * noise
    #                 ##clmap: if something is out of bound, it makes it -1 or 1 respectivelly
    #             audio = torch.clamp(audio, -1.0, 1.0)
    #     return audio

    def valid_loss(self, audio, audio_conditioner, phonemes_conditioner, Energy_conditioner, overlap = None):
        device = audio.device
        with torch.no_grad():
            N, T = audio.shape
            device = audio.device
            self.noise_level = self.noise_level.to(device)
            t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise
            predicted = self.model(noisy_audio, audio_conditioner.unsqueeze(1), t, phonemes_conditioner, Energy_conditioner)
            if overlap is None:
                loss_denoise = F.l1_loss(noise, predicted.squeeze(1))
            else:
                loss_denoise = self.get_loss_with_overlap_mask(noise, predicted, overlap)
            
            if self.params.spec_loss_coeff > 0.:
                loss_spec = F.l1_loss(self.log_mel_spec(noise), self.log_mel_spec(predicted.squeeze(1)))
                #logger.info(f'denoise loss: {loss_denoise:.4f} spec loss: {loss_spec:.4f}')
                loss_coeff = self.params.spec_loss_coeff
                loss = (1 - loss_coeff) * loss_denoise + loss_coeff * loss_spec
            else:
                loss_spec = None
                loss = loss_denoise
            
            return loss, loss_denoise, loss_spec

### TODO: if pseq / stoi  are relevant ###
    # def _audio_summary(self, step, n_samples=3):
    #     ##
    #     features = next(iter(self.valid_ds))
    #     clean, noisy = features[0].squeeze(1), features[1].squeeze(1)
    #     pred_audio = self.valid_step(noisy.to(self.device), self.params.valid_ds.n_samples).cpu()
    
    #     pred_mel = self.feature_extractor(pred_audio.cpu())
    #     clean_mel = self.feature_extractor(clean)
    #     noisy_mel = self.feature_extractor(noisy)

    #     for i in range(min(n_samples, pred_audio.shape[0])):
    #         print("Im in the loop")
    #         # self.summary_writer.add_audio(f'feature_{i}/pred_audio', pred_audio[i], step, sample_rate=self.params.sample_rate)
    #         # self.summary_writer.add_audio(f'feature_{i}/clean_audio', clean[i], step, sample_rate=self.params.sample_rate)
    #         # self.summary_writer.add_audio(f'feature_{i}/noisy_audio', noisy[i], step, sample_rate=self.params.sample_rate)
    #         # self.summary_writer.add_image(f'feature_{i}/clean_spec', plot_spectrogram_to_numpy(clean_mel[i].cpu().numpy()), step, dataformats="HWC")
    #         # self.summary_writer.add_image(f'feature_{i}/noisy_spec', plot_spectrogram_to_numpy(noisy_mel[i].cpu().numpy()), step, dataformats="HWC")
    #         # self.summary_writer.add_image(f'feature_{i}/pred_spec', plot_spectrogram_to_numpy(pred_mel[i].cpu().numpy()), step, dataformats="HWC")
    #     self.summary_writer.flush()
    #     print("finish audio aummery")

    def log_dict(self, d):
        if self.is_master:
            for k, v in d.items():
                self.summary_writer.add_scalar(k, v, self.step)
            self.summary_writer.flush()

def mytrain(replica_id, replica_count, port, params):
    torch.backends.cudnn.benchmark = True
    is_distributed = replica_count > 1
    params.noise_schedule = np.linspace(**params.noise_schedule,).tolist() # TODO ???????
    global logger
    logger.info = logger.info if replica_id == 0 else lambda x: x

    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
        device = torch.device('cuda', replica_id)
        torch.cuda.set_device(device)
        model = DiffAR(params).to(device)
        model = DistributedDataParallel(model, device_ids=[replica_id])
    else:
        model = DiffAR(params).cuda()
    if params.replica_id_attempt==3:
        replica_id=3
        device = torch.device('cuda', params.replica_id_attempt)
        torch.cuda.set_device(device)
        model = DiffAR(params).to(device)

    train_ds = instantiate(params.train_ds)
    train_ds = torch.utils.data.DataLoader(
        train_ds,
        batch_size=params.batch_size_train,
        shuffle=not is_distributed,
        num_workers=params.num_workers,
        sampler=DistributedSampler(train_ds) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
    )

    if 'valid_ds' in params:
        valid_ds = instantiate(params.valid_ds)
        valid_ds = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=params.batch_size_validation,
            shuffle=not is_distributed,
            num_workers=params.num_workers,
            sampler=DistributedSampler(valid_ds) if is_distributed else None,
            pin_memory=True,
            drop_last=True,
        )
    else:
        valid_ds = None

    if 'test_ds' in params:
        test_ds = instantiate(params.test_ds)
        test_ds = torch.utils.data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=not is_distributed,
            num_workers=params.num_workers,
            sampler=DistributedSampler(test_ds) if is_distributed else None,
            pin_memory=True,
            drop_last=True,
        )
    else:
        test_ds = None

    learner = DiffARLearner(model, train_ds, valid_ds, test_ds, (replica_id == 0), params, fp16=params.fp16)
    try:
        if params.test:
            learner.test()
        else:
            learner.train(max_steps=params.max_steps)
    except:
        print("stratin exept")
        torch.distributed.destroy_process_group()
