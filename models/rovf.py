import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel

# Load Perceiver and PerceiverV2 directly from RoVF-Meerkat-Reidentification
_ROVF_DIR = Path(__file__).resolve().parent.parent / "RoVF-Meerkat-Reidentification"
_PERCEIVER_PATH = _ROVF_DIR / "models" / "perceiver_wrapper.py"

_spec = importlib.util.spec_from_file_location("rovf_perceiver_wrapper", str(_PERCEIVER_PATH))
_perceiver_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_perceiver_mod)

Perceiver = _perceiver_mod.Perceiver
PerceiverV2 = _perceiver_mod.PerceiverV2


class RoVFVideoModel(nn.Module):
    """
    Exact implementation of RoVF (Recurrent Operations on Video Frames) from
    RoVF-Meerkat-Reidentification (RecurrentWrapper + Perceiver/PerceiverV2).

    Default configuration matches the best-performing variant: RoVF-S-ST
    (rovf_s_st_mask_no_aug_0p5_fps_meerkat.yml):
      - dino_model_name: 'facebook/dinov2-small'
      - freeze_image_model: True
      - model_type: 'recurrent_perceiverv2' (type_='v2')
      - is_append_avg_emb: True
      - embedding_dim: 384, latent_dim: 384, output_dim: 384
      - num_heads: 8, num_latents: 257, num_tf_layers: 2, dropout_rate: 0.1
      - num_frames: 5
    """

    def __init__(self, cfg=None):
        super().__init__()

        dino_model_name = getattr(cfg, "rovf_dino_model_name", "facebook/dinov2-small")
        dropout_rate = float(getattr(cfg, "rovf_dropout_rate", 0.1))
        freeze_image_model = bool(getattr(cfg, "rovf_freeze_image_model", True))
        is_append_avg_emb = bool(getattr(cfg, "rovf_is_append_avg_emb", True))
        perceiver_type = str(getattr(cfg, "rovf_perceiver_type", "v2")).lower()
        self.num_frames = int(getattr(cfg, "clip_len", 5) or 5)

        perceiver_config = {
            "raw_input_dim": None,
            "embedding_dim": int(getattr(cfg, "rovf_embedding_dim", 384)),
            "latent_dim": int(getattr(cfg, "rovf_latent_dim", 384)),
            "num_heads": int(getattr(cfg, "rovf_num_heads", 8)),
            "num_latents": int(getattr(cfg, "rovf_num_latents", 257)),
            "num_transformer_layers": int(getattr(cfg, "rovf_num_tf_layers", 2)),
            "dropout": dropout_rate,
            "output_dim": int(getattr(cfg, "rovf_output_dim", 384)),
            "use_raw_input": False,
            "use_embeddings": True,
            "flatten_channels": False,
        }

        self.recurrent_type = "perceiver"
        self.model_name = dino_model_name
        self.gru_linear = None

        if perceiver_type == "v1":
            self.recurrence_model = Perceiver(**perceiver_config)
        else:
            self.recurrence_model = PerceiverV2(**perceiver_config)

        # Load DINOv2-small backbone via HuggingFace AutoModel (exact match with RoVF)
        self.image_model = AutoModel.from_pretrained(dino_model_name, trust_remote_code=True)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.freeze_image_model = freeze_image_model
        self.is_append_avg_emb = is_append_avg_emb

        if self.freeze_image_model:
            for p in self.image_model.parameters():
                p.requires_grad = False
            self.image_model.eval()

        self.append_proj = None
        if is_append_avg_emb:
            emb_dim = perceiver_config.get("embedding_dim")
            out_dim = perceiver_config.get("output_dim")
            if out_dim is not None and emb_dim is not None and emb_dim != out_dim:
                self.append_proj = nn.Linear(emb_dim, out_dim)

        # RoVF trains and evaluates with raw Euclidean distance (TripletMarginLoss(margin=1.0, p=2))
        self.use_raw_euclidean = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_image_model:
            self.image_model.eval()
        return self

    def reset_latents(self):
        self.recurrence_model.reset_latents()

    def get_average(self, emb_list):
        stacked_tensors = torch.stack(emb_list, dim=1)
        stacked_tensors = self.dropout1(stacked_tensors)
        output_tensor = torch.mean(stacked_tensors, dim=-2)
        output_tensor = torch.mean(output_tensor, dim=-2)
        return output_tensor

    @torch.no_grad()
    def extract_frame_tokens(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract frozen DINOv2 last_hidden_state token sequences for a batch of clips.
        Args:
            video: FloatTensor of shape (B, T, C, H, W) in [0, 1]
        Returns:
            FloatTensor of shape (B, T, L, D) = (B, T, 257, 384)
        """
        self.image_model.eval()
        B, T, C, H, W = video.shape
        flat_frames = video.reshape(B * T, C, H, W)
        tokens = self.image_model(flat_frames).last_hidden_state
        return tokens.reshape(B, T, tokens.size(1), tokens.size(2)).detach()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting both:
          - Raw video clips: (B, T, C, H, W) or single images (B, C, H, W)
          - Pre-extracted DINOv2 token sequences: (B, T, L, D) where D=384
        """
        self.reset_latents()

        # Handle 4D tensor: either (B, C, H, W) image or (B, T, L, D) pre-extracted tokens
        if video.dim() == 4 and video.size(-1) == self.recurrence_model.embedding_dim and video.size(-2) == self.recurrence_model.num_latents:
            # Pre-extracted token sequence (B, T, 257, 384)
            if video.size(1) == 1 and self.num_frames > 1:
                video = video.repeat(1, self.num_frames, 1, 1)
            frame_list = []
            for i in range(video.size(1)):
                frame_output = video[:, i, :, :]
                if self.freeze_image_model:
                    frame_output = frame_output.detach()
                frame_output = self.dropout2(frame_output)
                frame_list.append(frame_output)
        else:
            if video.dim() == 4:
                # Single image (B, C, H, W) -> (B, 1, C, H, W)
                video = video.unsqueeze(1)
            # If evaluating single-frame query (T=1) on a multi-frame recurrent model, repeat to T=num_frames
            if video.size(1) == 1 and self.num_frames > 1:
                video = video.repeat(1, self.num_frames, 1, 1, 1)

            frame_list = []
            for i in range(video.size(1)):
                frame = video[:, i, :, :, :]
                if self.freeze_image_model:
                    with torch.no_grad():
                        frame_output = self.image_model(frame).last_hidden_state.detach()
                else:
                    frame_output = self.image_model(frame).last_hidden_state
                frame_output = self.dropout2(frame_output)
                frame_list.append(frame_output)

        stacked_tensors = torch.stack(frame_list, dim=1)
        avg_image_emb = torch.mean(stacked_tensors, dim=-3)

        prediction_list = []
        for i in range(len(frame_list)):
            pred = self.recurrence_model(
                raw_input=None,
                embeddings=frame_list[i],
                video_emb=avg_image_emb if i == 0 else None,
                is_reset_latents=False,
            )
            prediction_list.append(pred)
        self.reset_latents()

        if self.is_append_avg_emb:
            avg_emb = self.get_average(frame_list)
            if self.append_proj is not None:
                avg_emb = self.append_proj(avg_emb)
            return prediction_list[-1] + avg_emb

        return prediction_list[-1]
