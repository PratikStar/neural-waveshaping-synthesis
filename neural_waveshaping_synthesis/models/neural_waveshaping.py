import auraloss
import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os

from .modules.dynamic import TimeDistributedMLP
from .modules.generators import FIRNoiseSynth, HarmonicOscillator
from .modules.shaping import NEWT, Reverb
from ..utils.utils import HiddenPrints

gin.external_configurable(nn.GRU, module="torch.nn")
gin.external_configurable(nn.Conv1d, module="torch.nn")


@gin.configurable
class ControlModule(nn.Module):
    def __init__(self,
                 control_size: int,  # input to controlemodule
                 hidden_size,  # z
                 embedding_size: int,  # output of controlmodule
                 sample_rate: int,
                 control_hop: int,
                 z_dynamic_size: int = None,
                 z_static_size: int = None,
                 embedding_strategy: str = "NONE",  # NONE, GRU_LAST, FLATTEN_LINEAR, STATIC_DYNAMIC_Z
                 device: str = None
                 ):
        super().__init__()
        self.embedding_strategy = embedding_strategy
        self.sample_rate = sample_rate
        self.control_hop = control_hop

        # If sweeps is on, get hidden size from z_dim, else from gin hidden_size
        if 'WANDB_SWEEP_ID' in os.environ:
            print(f"ControlModule recognizes this is a sweep! hidden_size: {hidden_size}")
            self.z_static_size = hidden_size[0]
            self.z_dynamic_size = hidden_size[1]
        else:
            # print("Control module is NOT in sweep mode")
            self.z_static_size = z_static_size
            self.z_dynamic_size = z_dynamic_size
            self.hidden_size = self.z_dynamic_size + self.z_static_size
            print(f"self.z_static_size: {self.z_static_size}, self.z_dynamic_size: {self.z_dynamic_size}")

        if self.embedding_strategy in ["NONE", "GRU_LAST"]:
            self.gru = nn.GRU(control_size, hidden_size, batch_first=True)
            self.proj = nn.Conv1d(hidden_size, embedding_size, 1)

        elif self.embedding_strategy == "FLATTEN_LINEAR":
            self.gru = nn.GRU(control_size, hidden_size, batch_first=True)
            self.proj = nn.Conv1d(hidden_size, embedding_size, 1)
            self.flatten = nn.Flatten(1, 2)
            self.linear_encode = nn.Linear(hidden_size * (self.sample_rate // self.control_hop), hidden_size)
            self.con1d_decode = nn.Conv1d(1, self.sample_rate // self.control_hop,
                                          kernel_size=1)  # kernel size is hyperparam

        elif self.embedding_strategy == "STATIC_DYNAMIC_Z":
            # dynamic
            self.gru = nn.GRU(control_size, self.z_dynamic_size, batch_first=True)
            # static
            self.flatten = nn.Flatten(1, 2)
            self.linear_encode = nn.Linear(control_size * (self.sample_rate // self.control_hop), self.z_static_size)
            self.proj = nn.Conv1d(self.hidden_size, embedding_size, 1)
        elif self.embedding_strategy == "CONCAT_STATIC_Z":
            # dynamic
            self.embed = nn.Embedding(80, self.z_static_size)

            # self.timbre_z = {}
            # device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
            # for i in range(21):
            #     for a in ['A', 'B', 'C', 'D']:
            #         self.timbre_z[f"{i:02d}{a}"] = torch.nn.Parameter(torch.randn(self.z_static_size, device=device), requires_grad=True)
            print("Init: Here is the embedding table")
            print(self.embed)

            self.gru = nn.GRU(control_size + self.z_static_size, self.hidden_size, batch_first=True)
            # static
            self.proj = nn.Conv1d(self.hidden_size, embedding_size, 1)
        else:
            print("Please provide a correct embedding_strategy!!")

    def forward(self, x, presets=None):
        print(f"\nRunning ControlModule.forward")
        print(f"Embedding strategy: {self.embedding_strategy}")
        print(f"Input to control module: {x.shape}")
        print(f"Input presets to control module: {presets}")
        # print(f"presets embeddings: {self.timbre_z}")

        if self.embedding_strategy == "NONE":
            x, _ = self.gru(x.transpose(1, 2))
            print(f"After GRU: {x.shape}")
            print(x[0, 1, :10].detach().cpu().numpy())
            print(x[0, 2, :10].detach().cpu().numpy())
        if self.embedding_strategy == "GRU_LAST":
            x, _ = self.gru(x.transpose(1, 2))
            print(f"After GRU: {x.shape}")
            print(x[0, 1, :10].detach().cpu().numpy())
            print(x[0, 2, :10].detach().cpu().numpy())
            x_tmp = []
            for b in range(x.shape[0]):
                # print(f"batch: {b}")
                z = x[b, -1, :]
                # print(f"last embedding: {z}\n")
                z = z.repeat(x.shape[1], 1)
                x_tmp.append(z)
            x = torch.stack(x_tmp)
            print(f"GRU_LAST control embedding shape: {x.shape}")

            print(x[0, 1, :10].detach().cpu().numpy())
            print(x[0, 2, :10].detach().cpu().numpy())
            print(x[0, 3, :10].detach().cpu().numpy())
            # print(x[1,1,:10].detach().cpu().numpy())
            # print(x[1,2,:10].detach().cpu().numpy())
            # print(x[1,3,:10].detach().cpu().numpy())
        elif self.embedding_strategy == "FLATTEN_LINEAR":
            x, _ = self.gru(x.transpose(1, 2))
            print(f"After GRU: {x.shape}")
            print(x[0, 1, :10].detach().cpu().numpy())
            print(x[0, 2, :10].detach().cpu().numpy())

            flattened_x = self.flatten(x)
            flattened_x = flattened_x.unsqueeze(1)
            print(f"flattened_x: {flattened_x.shape}")

            z = self.linear_encode(flattened_x)
            print(f"z: {z.shape}")
            x = self.con1d_decode(z)
            print(f"con1d_decoded x: {x.shape}")
        elif self.embedding_strategy == "STATIC_DYNAMIC_Z":
            # dynamic
            z_dynamic, _ = self.gru(x.transpose(1, 2))
            print(f"After GRU (z_dynamic): {z_dynamic.shape}")
            print(z_dynamic[0, 0, :10].detach().cpu().numpy())
            print(z_dynamic[0, 1, :10].detach().cpu().numpy())
            # static z
            flattened_x = self.flatten(x).unsqueeze(1)
            print(f"flattened_x (for z_static): {flattened_x.shape}")

            z_static = self.linear_encode(flattened_x)
            print(f"z_static: {z_static.shape}")

            z_static = z_static.repeat(1, self.sample_rate // self.control_hop, 1)
            print(f"z_static after repeat: {z_static.shape}")
            print(z_static[0, 0, :10].detach().cpu().numpy())
            print(z_static[0, 1, :10].detach().cpu().numpy())

            x = torch.cat((z_dynamic, z_static), 2)
            print(f"After cat: {x.shape}")
            print(x[0, 0, :10].detach().cpu().numpy())
            print(x[0, 1, :10].detach().cpu().numpy())
        elif self.embedding_strategy == "CONCAT_STATIC_Z":
            # lookup
            z_static = self.embed(presets)
            print("After lookup")
            print(z_static)
            z_static = z_static.unsqueeze(1).repeat(1, self.sample_rate // self.control_hop, 1)
            print(f"after repeat: {z_static.shape}")
            print(z_static[0, 0, :10].detach().cpu().numpy())
            print(z_static[0, 1, :10].detach().cpu().numpy())
            x_cat = torch.cat((x.transpose(1, 2), z_static), 2)
            print(f"After cat: {x_cat.shape}")
            x_gru, _ = self.gru(x_cat)
            print(f"After GRU (y): {x_gru.shape}")
            y = self.proj(x_gru.transpose(1, 2))
            print(f"After Cond1D: {y.shape}")


            z_static_roll = torch.roll(z_static, 1, 0)
            print("After roll")
            print(z_static_roll)
            z_static_roll = z_static_roll.unsqueeze(1).repeat(1, self.sample_rate // self.control_hop, 1)
            print(f"after roll's repeat: {z_static_roll.shape}")
            print(z_static_roll[0, 0, :10].detach().cpu().numpy())
            print(z_static_roll[0, 1, :10].detach().cpu().numpy())
            x_cat_roll = torch.cat((x.transpose(1, 2), z_static_roll), 2)
            print(f"After cat roll: {x_cat_roll.shape}")
            x_gru_roll, _ = self.gru(x_cat_roll)
            print(f"After GRU roll (y): {x_gru_roll.shape}")
            y_roll = self.proj(x_gru_roll.transpose(1, 2))
            print(f"After Cond1D: {y_roll.shape}")

            return y, y_roll
        else:
            pass

        y = self.proj(x.transpose(1, 2))
        print(f"After Cond1D: {y.shape}")
        return y, x

    def get_control_from_z_(self, controls, z_static):
        print(f"\nRunning get_control_from_z_")
        print(f"Input to get_control_from_z_: {controls.shape}, {z_static.shape}")
        if self.embedding_strategy == "STATIC_DYNAMIC_Z":
            # dynamic
            z_dynamic, _ = self.gru(controls.transpose(1, 2))
            print(f"After GRU (z_dynamic): {z_dynamic.shape}")
            print(z_dynamic[0, 0, :10].detach().cpu().numpy())
            print(z_dynamic[0, 1, :10].detach().cpu().numpy())
            # static z will be provided
            # flattened_x = self.flatten(x).unsqueeze(1)
            # print(f"flattened_x (for z_static): {flattened_x.shape}")

            print(f"z_static: {z_static.shape}")

            z_static = z_static.repeat(1, self.sample_rate // self.control_hop, 1)
            print(f"z_static after repeat: {z_static.shape}")
            print(z_static[0, 0, :10].detach().cpu().numpy())
            print(z_static[0, 1, :10].detach().cpu().numpy())

            x = torch.cat((z_dynamic, z_static), 2)
            print(f"After cat: {x.shape}")
            print(x[0, 0, :10].detach().cpu().numpy())
            print(x[0, 1, :10].detach().cpu().numpy())
        elif self.embedding_strategy == "CONCAT_STATIC_Z":
            # lookup
            print(f"z_static: {z_static.shape}")
            z_static = z_static.unsqueeze(1).repeat(1, self.sample_rate // self.control_hop, 1)
            print(f"after repeat: {z_static.shape}")
            print(z_static[0, 0, :10].detach().cpu().numpy())
            print(z_static[0, 1, :10].detach().cpu().numpy())

            # concat
            x = torch.cat((controls.transpose(1, 2), z_static), 2)
            print(f"After cat: {x.shape}")

            x_gru, _ = self.gru(x)
            print(f"After GRU (y): {x_gru.shape}")

            y = self.proj(x_gru.transpose(1, 2))
            print(f"After Cond1D: {y.shape}")

            return y, x  # NOTE, because I need "x"

        else:
            pass

        y = self.proj(x.transpose(1, 2))
        print(f"After Cond1D: {y.shape}")
        return y, x


@gin.configurable
class NeuralWaveshaping(pl.LightningModule):
    def __init__(
            self,
            n_waveshapers: int,
            control_hop: int,
            sample_rate: float = 16000,
            learning_rate: float = 1e-3,
            lr_decay: float = 0.9,
            lr_decay_interval: int = 10000,
            log_audio: bool = True,
            hidden_size: [] = None,
            loss_roll_w: float = 0.01,
            finetune: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval
        self.control_hop = control_hop
        self.log_audio = log_audio
        self.loss_roll_w = loss_roll_w
        self.sample_rate = sample_rate

        self.embedding = ControlModule(hidden_size=hidden_size)

        self.osc = HarmonicOscillator()
        self.harmonic_mixer = nn.Conv1d(self.osc.n_harmonics, n_waveshapers, 1)

        self.newt = NEWT()

        with gin.config_scope("noise_synth"):
            self.h_generator = TimeDistributedMLP()
            self.noise_synth = FIRNoiseSynth()

        self.reverb = Reverb()

    def render_exciter(self, f0):
        print(f"\nCalling Harmonic oscillator with f0_upsampled[:, 0]")
        sig = self.osc(f0[:, 0])
        print(f"signal returned by harmonic oscillator: {sig.shape}")

        print(f"Calling Harmonic Mixer, applying Conv1d on harmonic spectrum")
        sig = self.harmonic_mixer(sig)
        print(f"sig: {sig.shape}")
        print(f"Returning from render_exciter\n")
        return sig

    def get_embedding(self, control, presets=None):
        print(f"\n In get_embedding")
        print(f"control: {control.shape}")
        print("splitting control in f0 and other")
        f0, other = control[:, 0:1], control[:, 1:2]
        print(f"f0: {f0.shape}")
        print(f"other: {other.shape}")
        control = torch.cat((f0, other), dim=1)
        print(f"concatenating f0 and other, control: {control.shape}")
        print(f"control: {control[0, 0, :10].detach().cpu().numpy()}")

        print("Invoking ControlModule with control")
        control_embedding, control_embedding_roll = self.embedding(control, presets)
        print(f"control_embedding: {control_embedding.shape}")
        print(f"control_embedding_roll: {control_embedding_roll.shape}")

        return control_embedding, control_embedding_roll

    def get_control_from_z(self, control, z):
        print(f"\n In get_control_from_z")
        print(f"control: {control.shape}")
        print("splitting control in f0 and other")
        f0, other = control[:, 0:1], control[:, 1:2]
        print(f"f0: {f0.shape}")
        print(f"other: {other.shape}")
        control = torch.cat((f0, other), dim=1)
        print(f"concatenating f0 and other, control: {control.shape}")
        print(f"control: {control[0, 0, :10].detach().cpu().numpy()}")

        print("Invoking ControlModule get_control_from_z_ with control and z")
        control_embedding, z = self.embedding.get_control_from_z_(control, z)
        print(f"control_embedding: {control_embedding.shape}")
        print(f"z: {z.shape}")

        return control_embedding, z

    def forward(self, f0, control, presets=None):  # control is 19 dimensional: 1f0, 1 loudness, 1 confidence, 16mfcc

        print(f"\n\n================= In forward ===================")
        print(f"f0: {f0.shape}")
        print(f"control: {control.shape}")
        print(f"preset: {presets}")
        print(f"f0: {f0[0, 0, :10].detach().cpu().numpy()}")

        pis = []
        for p in presets:
            i = (int(p[:2]) - 1) * 4
            s = int(ord(p[2])) - 64
            pis.append(i + s - 1)
        device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        presets = torch.tensor(pis).to(device)

        f0_upsampled = F.upsample(f0, f0.shape[-1] * self.control_hop,
                                  mode="linear")  # f0.shape[-1] is number of frames
        print(f"f0_upsampled: {f0_upsampled.shape}")
        print(f"f0_upsampled: {f0_upsampled[0, 0, :10].detach().cpu().numpy()}")

        x = self.render_exciter(f0_upsampled)
        print(f"x: {x.shape}")
        print(f"x: {x[0, 0, :10].detach().cpu().numpy()}")

        control_embedding, control_embedding_roll = self.get_embedding(control, presets=presets)

        ## Without ROLL
        print(f"\nInvoking NEWT with x and control_embedding")
        x_newt = self.newt(x, control_embedding)
        print(f"NEWT returns, x_newt: {x_newt.shape}")
        print("\nInvoking h_generator with control_embedding for noise synth")
        H = self.h_generator(control_embedding)
        print(f"H: {H.shape}")
        print("\nInvoking noise_synth")
        noise = self.noise_synth(H)
        print(f"noise: {noise.shape}")
        x_newt = torch.cat((x_newt, noise), dim=1)
        print(f"torch.cat((x_newt, noise), dim=1) -->: {x_newt.shape}")
        x_newt = x_newt.sum(1)
        print(f"x_newt.sum(1) -->: {x_newt.shape}")

        ## WITH ROLL
        print(f"\nInvoking NEWT with x and control_embedding")
        x_newt_roll = self.newt(x, control_embedding_roll)
        print(f"NEWT returns, x_newt_roll: {x_newt_roll.shape}")

        print("\nInvoking h_generator with control_embedding for noise synth")
        H_roll = self.h_generator(control_embedding_roll)
        print(f"H_roll: {H_roll.shape}")

        print("\nInvoking noise_synth")
        noise_roll = self.noise_synth(H_roll)
        print(f"noise_roll: {noise_roll.shape}")

        x_newt_roll = torch.cat((x_newt_roll, noise_roll), dim=1)
        print(f"torch.cat((x_newt_roll, noise_roll), dim=1) -->: {x_newt_roll.shape}")
        x_newt_roll = x_newt_roll.sum(1)
        print(f"x_newt_roll.sum(1) -->: {x_newt_roll.shape}")

        # print("Calling reverb")
        # x = self.reverb(x)
        # print(f"x: {x.shape}")
        return x_newt, x_newt_roll

    def encode(self, f0, control, presets):
        print(f"\n\n================= In Encode ===================")
        print(f"f0: {f0.shape}")
        print(f"control: {control.shape}")
        print(f"presets: {presets}")
        print(f"f0: {f0[0, 0, :10].detach().cpu().numpy()}")

        pis = []
        for p in presets:
            i = (int(p[:2]) - 1) * 4
            s = int(ord(p[2])) - 64
            pis.append(i + s - 1)
        device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        presets = torch.tensor(pis).to(device)

        f0_upsampled = F.upsample(f0, f0.shape[-1] * self.control_hop,
                                  mode="linear")  # f0.shape[-1] is number of frames
        print(f"f0_upsampled: {f0_upsampled.shape}")
        print(f"f0_upsampled: {f0_upsampled[0, 0, :10].detach().cpu().numpy()}")

        exciter_signal = self.render_exciter(f0_upsampled)
        print(f"x: {exciter_signal.shape}")
        print(f"x: {exciter_signal[0, 0, :10].detach().cpu().numpy()}")

        control_embedding, control_embedding_roll = self.get_embedding(control, presets=presets)
        return exciter_signal, control_embedding, control_embedding_roll

    def decode(self, exciter_signal, control_embedding, z):
        print(f"\n\n================= In Decode ===================")
        print(f"\nInvoking NEWT with exciter_signal and control_embedding")
        x = self.newt(exciter_signal, control_embedding)
        print(f"NEWT returns, x: {x.shape}")

        print("\nInvoking h_generator with control_embedding for noise synth")
        H = self.h_generator(control_embedding)
        print(f"H: {H.shape}")

        print("\nInvoking noise_synth")
        noise = self.noise_synth(H)
        print(f"noise: {noise.shape}")

        x = torch.cat((x, noise), dim=1)
        print(f"torch.cat((x, noise), dim=1) -->: {x.shape}")
        x = x.sum(1)
        print(f"x.sum(1) -->: {x.shape}")
        return x, z

    def configure_optimizers(self):
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.lr_decay_interval, self.lr_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _run_step(self, batch):
        audio = batch["audio"].float()
        f0 = batch["f0"].float()
        control = batch["control"].float()
        presets = [b[:3] for b in batch["name"]]

        audio_roll = torch.roll(audio, 1, 0)
        recon, recon_roll = self(f0, control,
                                    presets=presets
                                    )

        print(f"recon: {recon.shape}")
        print(f"audio: {audio.shape}")

        loss = self.stft_loss(recon, audio)
        loss_roll = self.stft_loss(recon_roll, audio_roll)
        total_loss = loss + self.loss_roll_w * loss_roll

        return loss, loss_roll, total_loss, recon, audio, recon_roll, audio_roll

    def _log_audio(self, name, audio):
        wandb.log(
            {
                "audio/%s"
                % name: wandb.Audio(audio, sample_rate=self.sample_rate, caption=f"name-{self.current_epoch}")
            },
            commit=False,
        )

    def training_step(self, batch, batch_idx):
        with HiddenPrints():
            loss, loss_roll, total_loss, _, _,_,_ = self._run_step(batch)
        self.log(
            "train/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/loss_roll",
            loss_roll.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/total_loss",
            total_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        loss, loss_roll, total_loss, recon, audio, recon_roll, audio_roll = self._run_step(batch)
        self.log(
            "val/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/loss_roll",
            loss_roll.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/total_loss",
            total_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if batch_idx == 0 and self.log_audio:
            self._log_audio(f"recon-og",
                            torch.cat((recon[0].detach().cpu().squeeze(),
                                       audio[0].detach().cpu().squeeze())))
            self._log_audio(f"recon-og-roll",
                            torch.cat((recon_roll[0].detach().cpu().squeeze(),
                                       audio_roll[0].detach().cpu().squeeze())))
        return total_loss

    def test_step(self, batch, batch_idx):
        loss, recon, audio = self._run_step(batch)
        self.log(
            "test/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if batch_idx == 0:
            self._log_audio("original", audio[0].detach().cpu().squeeze())
            self._log_audio("recon", recon[0].detach().cpu().squeeze())
