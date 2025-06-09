import torch
import numpy as np
import torch.nn as nn
import math
from experiments.robot.robot_utils import get_latent_action, get_model
from experiments.robot.openvla_utils import get_processor
from prismatic.models.policy.transformer_utils import MAPBlock


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class ActionDecoder(torch.nn.Module):
    def __init__(
        self,
        n_layers=1,
        vis_dim=4096,
        n_joints=16,
        window_size=30,
        hidden_dim=512,
        with_proprio=False,
        ):
        super().__init__()
        
        if with_proprio:
            self.proprio_proj = nn.Linear(n_joints, hidden_dim)
        self.latent_action_pool = MAPBlock(
            n_layers=n_layers,
            vis_dim=vis_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim//64,
            )
        self.visual_pool = MAPBlock(
            vis_dim=vis_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim//64,
            )

        if with_proprio:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 8), 
                nn.GELU(),
                nn.Linear(hidden_dim * 8, n_joints * window_size),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4), 
                nn.GELU(),
                nn.Linear(hidden_dim * 4, n_joints * window_size),
            )

    def forward(self, latent_action_tokens, visual_embed, proprio=None):
            
        visual_embed = self.visual_pool(visual_embed)
        
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)
        
        if proprio is not None:
            proprio = proprio.squeeze(1)
            proprio = self.proprio_proj(proprio)
            action = self.proj(torch.cat((action_token, proprio), dim=1))
        else:
            action = self.proj(action_token)

        return action


class ActionDecoderWrapper(nn.Module):
    def __init__(
        self,
        window_size=30,
        n_layers=1,
        hidden_dim=512,
        n_joints=16,
        balancing_factor=0.01,
        with_proprio=False,
        ):
        super().__init__()
        self.net = ActionDecoder(
            n_layers=n_layers,
            window_size=window_size,
            hidden_dim=hidden_dim,
            n_joints=n_joints,
            with_proprio=with_proprio,
            )
        
        self.with_proprio = with_proprio
        self.n_joints = n_joints
        self.temporal_size = window_size
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], n_joints))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        # Action chunking with temporal aggregation
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]


    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], self.n_joints))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

    
    def forward(self, latent_actions, visual_embed, proprio=None):
        
        # Run specialist policy
        if self.with_proprio:
            proprio = proprio.to(torch.float)
        else:
            proprio = None
        
        # Forward action decoder
        pred_action = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float), proprio).reshape(-1, self.temporal_size, self.n_joints)
        pred_action = np.array(pred_action.tolist())
        
        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)

        # gripper action clip
        
        if action_prediction[-1] < 0.5:
            action_prediction[-1] = 0
        else:
            action_prediction[-1] = 1


        if action_prediction[7] < 0.5:
            action_prediction[7] = 0
        else:
            action_prediction[7] = 1
            
        return action_prediction


class WrappedModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load action decoder
        self.action_decoder = ActionDecoderWrapper(
            window_size=cfg.window_size,
            n_layers=cfg.n_layers,
            hidden_dim=cfg.hidden_dim,
            balancing_factor=cfg.balancing_factor,
            with_proprio=cfg.with_proprio,
            )
        
        self.action_decoder.net.load_state_dict(torch.load(cfg.action_decoder_path))

        # Load VLA
        self.vla = get_model(cfg)


class WrappedGenieEvaluation():
    def __init__(self, cfg, wrapped_model):
        super().__init__()
        self.cfg = cfg

        self.model = wrapped_model
        # [OpenVLA] Get Hugging Face processor
        self.processor = get_processor(cfg)

        self.prev_hist_action = ['']

        

    def reset(self,):
        """
        This is called
        """ 
        self.model.module.action_decoder.reset()
        self.prev_hist_action = ['']


    def step(self, img_h, img_l, img_r, lang, proprio=np.zeros(16)):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """

        observation = {
            "img_h": img_h,
            "img_l": img_l,
            "img_r": img_r,
            "state": [],
        }
        
        start_idx = len(self.prev_hist_action) if len(self.prev_hist_action) < 4 else 4
        prompt_hist_action_list = [self.prev_hist_action[idx] for idx in range(-1 * start_idx, 0)]
        prompt_hist_action = ''
        for latent_action in prompt_hist_action_list:
            prompt_hist_action += latent_action

        # Query model to get latent action
        latent_action, visual_embed, generated_ids = get_latent_action(
            self.cfg,
            self.model.vla,
            observation,
            lang,
            processor=self.processor,
            hist_action=self.prev_hist_action[-1],
        )

        latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]
        hist_action = ''
        all_correct = True
        for latent_action_ids in generated_ids[0]:
            if latent_action_ids.item() - 32001 > 31:
                all_correct = False
                break
            hist_action += latent_action_detokenize[latent_action_ids.item() - 32001]
        if all_correct:
            self.prev_hist_action.append(hist_action)

        # Get proprio signal
        state = torch.from_numpy(proprio).to(latent_action.device, dtype=torch.float)
        state = state.unsqueeze(0)

        # Get decoded action
        action = self.model.action_decoder(latent_action, visual_embed, state)

        return action