import auraloss
import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .modules.dynamic import TimeDistributedMLP
from .modules.generators import FIRNoiseSynth, HarmonicOscillator
from .modules.shaping import NEWT, Reverb

gin.external_configurable(nn.GRU, module="torch.nn")
gin.external_configurable(nn.Conv1d, module="torch.nn")


@gin.configurable
class ControlModule(nn.Module):
    def __init__(self, control_size: int, hidden_size: int, embedding_size: int, embedding_strategy: str = "NONE"):
        super().__init__()
        self.embedding_strategy = embedding_strategy
        self.gru = nn.GRU(control_size, hidden_size, batch_first=True)
        self.proj = nn.Conv1d(hidden_size, embedding_size, 1)

    def forward(self, x):
        print(f"\nRunning ControlModule.forward")
        print(f"Embedding strategy: {self.embedding_strategy}")
        print(f"Input to control module: {x.shape}")
        # print(f"before GRU: {x[0,1,:10].detach().cpu().numpy()}")
        x, _ = self.gru(x.transpose(1, 2))
        print(f"After GRU: {x.shape}")
        print(x[0,1,:10].detach().cpu().numpy())
        print(x[0,2,:10].detach().cpu().numpy())
        print(x[0,3,:10].detach().cpu().numpy())

        if self.embedding_strategy == "GRU_LAST":
            x_tmp = []
            for b in range(x.shape[0]):
                print(f"batch: {b}")
                z = x[b, -1, :]
                # print(f"last embedding: {z}\n")
                z = z.repeat(x.shape[1], 1)
                x_tmp.append(z)
            x = torch.stack(x_tmp)
            print(f"GRU_LAST control embedding shape: {x.shape}")

            print(x[0,1,:10].detach().cpu().numpy())
            print(x[0,2,:10].detach().cpu().numpy())
            print(x[0,3,:10].detach().cpu().numpy())
            # print(x[1,1,:10].detach().cpu().numpy())
            # print(x[1,2,:10].detach().cpu().numpy())
            # print(x[1,3,:10].detach().cpu().numpy())
        else:
            pass

        y = self.proj(x.transpose(1, 2))
        print(f"{y.shape}")
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
        log_audio: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval
        self.control_hop = control_hop
        self.log_audio = log_audio

        self.sample_rate = sample_rate

        self.embedding = ControlModule()

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

    def get_embedding(self, control):
        print(f"\n In get_embedding")
        print(f"control: {control.shape}")
        print("splitting control in f0 and other")
        f0, other = control[:, 0:1], control[:, 1:2]
        print(f"f0: {f0.shape}")
        print(f"other: {other.shape}")
        control = torch.cat((f0, other), dim=1)
        print(f"concatenating f0 and other, control: {control.shape}")
        print(f"control: {control[0,0,:10].detach().cpu().numpy()}")

        print("Invoking ControlModule with control")
        control_embedding, gru_embedding = self.embedding(control)
        print(f"control_embedding: {control_embedding.shape}")
        print(f"gru_embedding: {gru_embedding.shape}")

        return control_embedding, gru_embedding

    def forward(self, f0, control): # control is 19 dimensional: 1f0, 1 loudness, 1 confidence, 16mfcc
        print(f"\n\n================= In forward ===================")
        print(f"f0: {f0.shape}")
        print(f"control: {control.shape}")
        print(f"f0: {f0[0,0,:10].detach().cpu().numpy()}")

        f0_upsampled = F.upsample(f0, f0.shape[-1] * self.control_hop, mode="linear") # f0.shape[-1] is number of frames
        print(f"f0_upsampled: {f0_upsampled.shape}")
        print(f"f0_upsampled: {f0_upsampled[0,0,:10].detach().cpu().numpy()}")

        x = self.render_exciter(f0_upsampled)
        print(f"x: {x.shape}")
        print(f"x: {x[0,0,:10].detach().cpu().numpy()}")

        control_embedding, gru_embedding = self.get_embedding(control)
        print(f"control_embedding: {control_embedding[0,:10,0].detach().cpu().numpy()}")
        print(f"control_embedding: {control_embedding[0,:10,1].detach().cpu().numpy()}")
        print(f"control_embedding: {control_embedding[0,:10,2].detach().cpu().numpy()}")
        print(f"control_embedding: {control_embedding[0,:10,3].detach().cpu().numpy()}")

        print(f"\nInvoking NEWT with x and control_embedding")
        x = self.newt(x, control_embedding)
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

        # print("Calling reverb")
        # x = self.reverb(x)
        # print(f"x: {x.shape}")

        return x, gru_embedding

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

        recon, gru_embedding = self(f0, control)

        print(f"recon: {recon.shape}")
        print(f"audio: {audio.shape}")

        loss = self.stft_loss(recon, audio)
        return loss, recon, audio

    def _log_audio(self, name, audio):
        wandb.log(
            {
                "audio/%s"
                % name: wandb.Audio(audio, sample_rate=self.sample_rate, caption=name)
            },
            commit=False,
        )

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._run_step(batch)
        self.log(
            "train/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, audio = self._run_step(batch)
        self.log(
            "val/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if batch_idx == 0 and self.log_audio:
            self._log_audio("original", audio[0].detach().cpu().squeeze())
            self._log_audio("recon", recon[0].detach().cpu().squeeze())
        return loss

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
