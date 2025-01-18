from datetime import datetime
import os
import numpy as np
import pygame
import torch
from pathlib import Path
from huggingface_hub import snapshot_download as hf_snapshot_download
from torchvision import transforms
from PIL import Image
import itertools
import random

from dweam import Game, GameInfo, Field, get_cache_dir

from .models.gen.edm import EDM
from .models.gen.blocks import UNet

REPO_ID = "juramoshkov/snake-diffusion"

def snapshot_download(**kwargs) -> Path:
    """Download and cache files from Hugging Face."""
    base_cache_dir = get_cache_dir()
    cache_dir = base_cache_dir / 'huggingface-data'
    path = hf_snapshot_download(cache_dir=str(cache_dir), **kwargs)
    return Path(path)

class SnakeDiffusionGame(Game):
    class Params(Game.Params):
        denoising_steps: int = Field(default=10, description="Number of denoising steps")
        context_window: int = Field(
            default=4, 
            ge=1,
            le=4,
            description="Number of previous frames to consider",
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():  # Support for Mac M1/M2
            self.device = torch.device("mps")
            
        # Model parameters from notebook
        input_channels = 3
        actions_count = 5  # 4 directions + initial state
        
        # Use the same resolution as the training notebook
        self.width = 64
        self.height = 64
        
        # Load training data
        training_dir = Path(__file__).parent / "training_data"
        if not training_dir.exists():
            raise FileNotFoundError(f"Training data directory not found at {training_dir}")
            
        # Load and sort snapshot frames
        snapshots_dir = training_dir / "snapshots"
        if not snapshots_dir.exists():
            raise FileNotFoundError(f"Snapshots directory not found at {snapshots_dir}")
            
        self.snapshot_frames = sorted(
            snapshots_dir.glob("*.jpg"), 
            key=lambda p: int(p.stem)
        )
        
        # Load actions file
        actions_file = training_dir / "actions"
        if not actions_file.exists():
            raise FileNotFoundError(f"Actions file not found at {actions_file}")
            
        # Read actions file (one number per line)
        with open(actions_file) as f:
            self.actions_data = [int(line.strip()) for line in f]
        
        # Verify matching lengths
        if len(self.snapshot_frames) != len(self.actions_data):
            raise ValueError(
                f"Mismatch between number of frames ({len(self.snapshot_frames)}) "
                f"and actions ({len(self.actions_data)})"
            )
            
        # Initialize transform for loading images
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5), (.5,.5,.5))
        ])
        
        try:
            # Download model files from HuggingFace
            self.log.info("Downloading files from HuggingFace", repo_id=REPO_ID)
            model_dir = snapshot_download(repo_id=REPO_ID)
            
            # Initialize EDM model
            self.model = EDM(
                p_mean=-1.2,
                p_std=1.2,
                sigma_data=0.5,
                model=UNet(
                    (input_channels) * (self.params.context_window + 1),
                    3,
                    None,
                    actions_count,
                    self.params.context_window
                ),
                context_length=self.params.context_window,
                device=self.device
            )
            
            # Load model weights
            model_path = Path(model_dir) / "model.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.log.info("Loading model weights", path=str(model_path))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model"])
            self.model.eval()
            self.log.info("Model loaded successfully")
            
        except Exception:
            self.log.exception("Failed to initialize game")
            raise

        # Game state initialization
        self.action_states = {
            "up": False,
            "down": False,
            "left": False,
            "right": False
        }
        
        self.current_direction = "right"
        self.frame_count = 0
        
        # Initialize with random sequence
        self.reset()

    def step(self) -> pygame.Surface:
        # Process current input state
        action = self.get_current_action()
        
        # Generate new frame
        with torch.no_grad():
            # Create masked versions of history tensors
            masked_frames = self.frame_history.clone()
            masked_actions = self.action_history.clone()
            
            # Zero out frames beyond the current context window
            if self.params.context_window < 4:  # 4 is the max context window
                masked_frames[:-self.params.context_window] = 0
                masked_actions[:-self.params.context_window] = 0
            
            frame = self.model.sample(
                self.params.denoising_steps,
                (3, self.height, self.width),
                masked_frames.unsqueeze(0),
                masked_actions.unsqueeze(0)
            )[0]
        
        # Update history - always maintain 4 frames
        self.frame_history = torch.cat([self.frame_history[1:], frame.unsqueeze(0)], dim=0)
        self.action_history = torch.cat([
            self.action_history[1:],
            torch.tensor([action], dtype=torch.long, device=self.device)
        ], dim=0)
        
        self.frame_count += 1
        
        # Convert to pygame surface
        return self.display_frame(frame)

    def get_current_action(self) -> int:
        # Convert action states to model's action space
        # Actions: [right, left, up, down]
        direction = None
        for key in ["up", "down", "left", "right"]:
            if self.action_states[key]:
                if ((key == "up" and self.current_direction != "down") or
                    (key == "down" and self.current_direction != "up") or
                    (key == "left" and self.current_direction != "right") or
                    (key == "right" and self.current_direction != "left")):
                    direction = key
                    self.current_direction = key
                break
        
        # Convert direction to action index
        if direction == "right":
            return 0
        elif direction == "left":
            return 1
        elif direction == "up":
            return 2
        elif direction == "down":
            return 3
        else:
            # If no valid direction, continue in current direction
            if self.current_direction == "right":
                return 0
            elif self.current_direction == "left":
                return 1
            elif self.current_direction == "up":
                return 2
            else:  # down
                return 3

    def generate_frame(self, action: int) -> torch.Tensor:
        try:
            # We now always have exactly CONTEXT_WINDOW frames/actions
            frame = self.model.sample(
                self.params.denoising_steps,
                (3, self.height, self.width),
                self.frame_history.unsqueeze(0),
                self.action_history.unsqueeze(0)
            )[0]
            return frame
        except Exception:
            self.log.exception("Error in diffusion generation")
            return torch.zeros((3, self.height, self.width), device=self.device)

    def display_frame(self, frame: torch.Tensor) -> pygame.Surface:
        # Convert to numpy and scale to [0, 255]
        frame = (frame * 127.5 + 127.5).long().clip(0, 255)
        frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # BGR to RGB if needed
        if frame.shape[-1] == 3:
            frame = frame[..., ::-1]
        
        return pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    def on_key_down(self, key: int) -> None:
        if key == pygame.K_w or key == pygame.K_UP:
            self.action_states["up"] = True
        elif key == pygame.K_s or key == pygame.K_DOWN:
            self.action_states["down"] = True
        elif key == pygame.K_a or key == pygame.K_LEFT:
            self.action_states["left"] = True
        elif key == pygame.K_d or key == pygame.K_RIGHT:
            self.action_states["right"] = True
        elif key == pygame.K_r or key == pygame.K_RETURN:
            self.reset()

    def on_key_up(self, key: int) -> None:
        if key == pygame.K_w or key == pygame.K_UP:
            self.action_states["up"] = False
        elif key == pygame.K_s or key == pygame.K_DOWN:
            self.action_states["down"] = False
        elif key == pygame.K_a or key == pygame.K_LEFT:
            self.action_states["left"] = False
        elif key == pygame.K_d or key == pygame.K_RIGHT:
            self.action_states["right"] = False


    def stop(self) -> None:
        """Clean up GPU resources when stopping the game"""
        super().stop()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove model references
        self.model = None
        self.frame_history = None
        self.action_history = None

    def reset(self) -> None:
        """Reset the game state with random sequence from training data."""
        # Reset action states
        self.action_states = {
            "up": False,
            "down": False,
            "left": False,
            "right": False
        }
        
        # Reset direction to initial state
        self.current_direction = "right"
        
        # Reset frame count
        self.frame_count = 0
        
        if self.device is not None:  # Check if game hasn't been stopped
            # Pick random starting index (ensuring we have 4 consecutive frames)
            max_start_idx = len(self.snapshot_frames) - 4
            start_idx = random.randint(0, max_start_idx)
            
            # Load 4 consecutive frames
            frames = []
            for idx in range(start_idx, start_idx + 4):
                img = Image.open(self.snapshot_frames[idx])
                tensor = self.transform(img)
                frames.append(tensor)
            
            self.frame_history = torch.stack(frames).to(self.device)
            
            # Get corresponding actions
            self.action_history = torch.tensor(
                self.actions_data[start_idx:start_idx + 4],
                dtype=torch.long,
                device=self.device
            )

