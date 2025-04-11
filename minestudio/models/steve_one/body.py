from copy import deepcopy
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from minestudio.models.base_policy import MinePolicy
from huggingface_hub import PyTorchModelHubMixin

from minestudio.utils.mineclip_lib.mineclip import MineCLIP
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.models.base_policy import MinePolicy

class TranslatorVAE(torch.nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256, latent_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, visual_embeddings, text_embeddings):
        """Encode the given visual and text embeddings into a latent vector."""
        # Concatenate the visual and text embeddings.
        x = torch.cat([visual_embeddings, text_embeddings], dim=1)
        # Encode the concatenated embeddings into a latent vector.
        return self.encoder(x)

    def sample(self, mu, logvar):
        """Sample a latent vector from the given mu and logvar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_vector, text_embeddings):
        """Decode the given latent vector and text embeddings into a visual embedding."""
        # Concatenate the latent vector and text embeddings.
        x = torch.cat([latent_vector, text_embeddings], dim=1)
        # Decode the concatenated embeddings into a visual embedding.
        return self.decoder(x)

    def forward(self, text_embeddings, deterministic=False):
        """Encode the given text embeddings into a latent vector and then decode it into a visual embedding."""
        # Use the prior as the mean and logvar.
        mu = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)
        logvar = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)

        # Sample a latent vector from the mu and logvar.
        if deterministic:
            latent_vector = mu
        else:
            latent_vector = self.sample(mu, logvar)

        # Decode the latent vector into a visual embedding.
        pred_visual_embeddings = self.decode(latent_vector, text_embeddings)

        return pred_visual_embeddings

class ImgPreprocessing(nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(torch.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = nn.Parameter(torch.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        x = img
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x

class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        return self.linear(self.cnn(img))

class MinecraftPolicy(nn.Module):
    """
    :param recurrence_type:
        None                - No recurrence, adds no extra layers
        lstm                - (Depreciated). Singular LSTM
        multi_layer_lstm    - Multi-layer LSTM. Uses n_recurrence_layers to determine number of consecututive LSTMs
            Does NOT support ragged batching
        multi_masked_lstm   - Multi-layer LSTM that supports ragged batching via the first vector. This model is slower
            Uses n_recurrence_layers to determine number of consecututive LSTMs
        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(
        self,
        recurrence_type="transformer", ## 
        impala_width=1,
        impala_chans=(16, 32, 32),
        obs_processing_width=256,
        hidsize=512,
        single_output=False,  # True if we don't need separate outputs for action/value outputs
        img_shape=None,
        scale_input_img=True,
        only_img_input=False,
        init_norm_kwargs={},
        impala_kwargs={},
        # Unused argument assumed by forc.
        input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False,
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,  # Not needed for transformer
        mineclip_embed_dim=512,  # MODIFIED (added this)
        **unused_kwargs,
    ):
        super().__init__()
        # import pdb
        # pdb.set_trace()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        active_reward_monitors = active_reward_monitors or {}

        self.single_output = single_output

        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.hidsize = hidsize

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # Setup inputs
        self.img_preprocess = ImgPreprocessing(img_statistics=img_statistics, scale_img=scale_input_img)
        self.img_process = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs,
        )

        self.pre_lstm_ln = nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None

        self.recurrence_type = recurrence_type

        self.recurrent_layer = None
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        )

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = torch.nn.LayerNorm(hidsize)

        # MODIFIED (added this)
        self.mineclip_embed_linear = torch.nn.Linear(mineclip_embed_dim, hidsize)

    def output_latent_size(self):
        return self.hidsize

    def forward(self, ob, state_in, context):
        b, t = ob["img"].shape[:2]
        first = context["first"].bool()

        x = self.img_preprocess(ob["img"])
        x = self.img_process(x)

        if self.diff_obs_process:
            processed_obs = self.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        # MODIFIED (added this)
        mineclip_embed = ob["mineclip_embed"].reshape(b * t, -1)
        # Normalize mineclip_embed (doesn't work because the norm is way too small then?)
        # mineclip_embed = F.normalize(mineclip_embed, dim=-1)
        mineclip_embed = self.mineclip_embed_linear(mineclip_embed)
        mineclip_embed = mineclip_embed.reshape(b, t, -1)
        x = x + mineclip_embed

        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        else:
            state_out = state_in

        x = F.relu(x, inplace=False)

        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, state_out
        return (pi_latent, vf_latent), state_out

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None

class SteveOnePolicy(MinePolicy, PyTorchModelHubMixin):
    
    def __init__(
        self, 
        mineclip_kwargs: dict = {},
        prior_kwargs: dict = {},
        policy_kwargs: dict = {},
        freeze_mineclip: bool = True,
        action_space = None,
    ):
        net = MinecraftPolicy(** policy_kwargs)
        super().__init__(hiddim=net.hidsize, action_space=action_space)

        self.net = net
        self.prior = TranslatorVAE(** prior_kwargs)
        self.mineclip = MineCLIP(** mineclip_kwargs)

        if freeze_mineclip:
            for param in self.mineclip.parameters():
                param.requires_grad = False

    def prepare_condition(self, instruction: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        assert 'cond_scale' in instruction, "instruction must have 'cond_scale' key."

        if 'video' in instruction:
            assert 'text' not in instruction, "cannot have both text and video in instruction"
            
            video = instruction['video']

            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video).to(self.device)
            if video.dim() == 4:
                video = rearrange(video, 'T H W C -> 1 T C H W')
            if video.shape[2] != 3:
                video = rearrange(video, 'B T H W C -> B T C H W')

            assert video.dim() == 5 and video.shape[2] == 3, "video must be a 5D tensor with shape (B, T, C, H, W) or (B, T, H, W, C)"
            B, T, C, H, W = video.shape

            if video.dtype == torch.uint8:
                mineclip_inputs = video.float()
            elif video.dtype == torch.float32:
                assert video.abs().max() <= 1.0, "float32 video must be in range [-1, 1]"
                mineclip_inputs = video * 255.0
            else:
                raise ValueError("video must be either uint8 or float32.")
            
            mineclip_inputs = rearrange(
                torch.nn.functional.interpolate(
                    rearrange(mineclip_inputs, 'B T C H W -> (B T) C H W'),
                    size=(160, 256), 
                    mode='bilinear', 
                    align_corners=False
                ),
                '(B T) C H W -> B T C H W',
                B=B, T=T
            )

            mineclip_embeds = self.mineclip.encode_video(mineclip_inputs)
        else:
            assert 'text' in instruction, "instruction must have either text or video."

            texts = instruction['text']
            if isinstance(texts, str):
                texts = [texts]
            assert isinstance(texts, list) and isinstance(texts[0], str), "text must be a string or a list of strings."
            
            text_embeds = self.mineclip.encode_text(texts)
            mineclip_embeds = self.prior(text_embeds, deterministic=deterministic)

        return {
            'cond_scale': instruction['cond_scale'],
            'mineclip_embeds': mineclip_embeds,
        }

    def forward(
        self, 
        condition: Dict[str, Any], #
        input: Dict[str, Any], 
        state_in: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            latents: containing `pi_logits` and `vpred` latent tensors.
            state_out: containing the updated state tensors.
        """
        input, condition, state_in = input.copy(), condition.copy(), state_in.copy()
        mineclip_embeds = condition['mineclip_embeds']
        if mineclip_embeds.shape[0] == 1 and input['image'].shape[0] > 1:
            mineclip_embeds = repeat(mineclip_embeds, '1 ... -> b ...', b=input['image'].shape[0])
        if condition['cond_scale'] != 0 and condition['cond_scale'] is not None:
            state_in = [rearrange(x, "b ... c -> (b c) ...") for x in state_in]
            images = repeat(input['image'], "b ... -> (b c) ...", c=2)
            mineclip_embeds = rearrange(
                torch.stack([mineclip_embeds, torch.zeros_like(mineclip_embeds)]),
                'c b ... -> (b c) ...'
            )
        else:
            images = input['image']

        dummy_first = torch.zeros((images.shape[0], images.shape[1]), dtype=torch.bool, device=self.device)

        if images.shape[-1] != 3:
            images = rearrange(images, 'b t c h w -> b t h w c')
        if images.dtype == torch.uint8:
            images = images.float()
        elif images.dtype == torch.float32:
            assert images.abs().max() <= 1.0, "float32 image must be in range [-1, 1]"
            images = images * 255.0
        else:
            raise ValueError("image must be either uint8 or float32.")

        (pi_latent, vf_latent), state_out = self.net(
            ob={"img": images, "mineclip_embed": repeat(mineclip_embeds, 'b c -> b t c', t=images.shape[1])}, 
            context={"first": dummy_first}, 
            state_in=state_in
        )

        pi_logits = self.pi_head(pi_latent)
        vpred = self.value_head(vf_latent)
        if condition['cond_scale'] != 0 and condition['cond_scale'] is not None:
            pi_logits = {k: rearrange(v, '(b c) ... -> b c ...', c=2) for k, v in pi_logits.items()}
            vpred = rearrange(vpred, '(b c) ... -> b c ...', c=2)
            state_out = [rearrange(x, '(b c) ... -> b ... c', c=2) for x in state_out]
            
            pi_logits = {k: (1 + condition['cond_scale']) * v[:, 0] - condition['cond_scale'] * v[:, 1] for k, v in pi_logits.items()}
            vpred = vpred[:, 0]

        latents = {
            "pi_logits": pi_logits, 
            "vpred": vpred, 
        }
        return latents, state_out
    
    def initial_state(
        self, 
        condition: Dict[str, Any], 
        batch_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        initial_state = self.net.initial_state(batch_size)
        if condition['cond_scale'] == 0.0 or condition['cond_scale'] is None:
            return initial_state
        else:
            return [torch.stack([x, x], dim=-1) for x in initial_state]

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

def load_steve_one_policy(ckpt_path: str) -> SteveOnePolicy:
    return SteveOnePolicy.from_pretrained(ckpt_path)

if __name__ == '__main__':
    model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to("cuda")
    model.eval()
    condition = model.prepare_condition(
        {
            'cond_scale': 4.0,
            'video': np.random.randint(0, 255, (2, 16, 224, 224, 3)).astype(np.uint8)
        }
    )

    output, memory = model(condition,
        input={
            'image': torch.zeros(2, 8, 128, 128, 3).to("cuda"), 
        },
        state_in=model.initial_state(condition, 2)
    )
    import pdb
    pdb.set_trace()