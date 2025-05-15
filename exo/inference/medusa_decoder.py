"""
Medusa model adapter and decoder for faster LLM inference.

This module provides both:
1. MedusaModel - A wrapper that adds Medusa heads to existing models 
2. MedusaDecoder - The decoder algorithm with probability threshold filtering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import os
import json
import time
import numpy as np

logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    """
    A Residual Block module for Medusa heads.
    
    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the LLaMA model
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass of the ResBlock."""
        return x + self.act(self.linear(x))

class MedusaHead(nn.Module):
    """
    A Medusa head that predicts tokens at specific future positions.
    """
    def __init__(self, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.layers = nn.Sequential(
            *([ResBlock(hidden_size)] * num_layers),
            nn.Linear(hidden_size, vocab_size, bias=False)
        )
        
    def forward(self, x):
        """Forward pass of the Medusa head."""
        return self.layers(x)

class MedusaModel(nn.Module):
    """
    MedusaModel adapter that adds multiple prediction heads to a base model.
    """
    def __init__(
        self,
        base_model,
        hidden_size,
        vocab_size,
        medusa_num_heads=4,
        medusa_num_layers=1
    ):
        """
        Initialize the MedusaModel adapter.
        
        Args:
            base_model: The underlying model to add Medusa heads to
            hidden_size: Hidden size of the model
            vocab_size: Vocabulary size of the model
            medusa_num_heads: Number of Medusa heads to add
            medusa_num_layers: Number of layers in each Medusa head
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        
        # Create Medusa heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size, medusa_num_layers)
            for _ in range(medusa_num_heads)
        ])
        
        # Tree attention state
        self.medusa_mode = False
        self.medusa_mask = None
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs
    ):
        """
        Forward pass of the MedusaModel.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Cached key values for faster inference
            output_orig: Whether to output the original logits
            position_ids: Position IDs
            medusa_forward: Whether to use Medusa forward pass
            **kwargs: Additional arguments to pass to the base model
            
        Returns:
            Tuple of (medusa_logits, outputs, logits) if output_orig is True,
            otherwise just medusa_logits
        """
        # Only set medusa_mode and mask if medusa_forward is True
        if medusa_forward:
            self.medusa_mode = True
            
        # Forward pass through the base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **kwargs
        )
        
        # Get the last hidden states
        last_hidden_state = outputs.last_hidden_state
        
        # Apply Medusa heads to get logits
        medusa_logits = [head(last_hidden_state) for head in self.medusa_heads]
        
        # Original model logits
        logits = outputs.logits
        
        # Reset medusa_mode after forward pass
        self.medusa_mode = False
        
        if output_orig:
            return medusa_logits, outputs, logits
        else:
            return medusa_logits
    
    def get_tokenizer(self):
        """Get the tokenizer from the base model."""
        return getattr(self.base_model, "tokenizer", None)
    
    def generate(self, *args, **kwargs):
        """Use our modified MedusaDecoder for generation with probability threshold filtering.
        
        This intercepts normal generation to apply our probability filtering.
        """
        from exo.inference.medusa_decoder import MedusaDecoder
        import os
        
        # Print clear indicator that our modified code is running
        print("\n!!!!! USING CUSTOM MEDUSA DECODER WITH PROBABILITY THRESHOLD !!!!!")
        
        # Extract relevant parameters
        tokenizer = self.get_tokenizer()
        
        # Check for probability threshold in environment or kwargs
        probability_threshold = kwargs.pop('probability_threshold', 0.5)  # Default 0.5
        env_prob_threshold = os.environ.get('MEDUSA_PROBABILITY_THRESHOLD')
        if env_prob_threshold is not None:
            try:
                probability_threshold = float(env_prob_threshold)
            except ValueError:
                pass
        
        print(f"Probability threshold: {probability_threshold}")
        
        # Check for debug mode
        debug = kwargs.pop('debug', False)
        env_debug = os.environ.get('MEDUSA_DEBUG', '').lower() in ('true', '1', 't', 'yes')
        debug = debug or env_debug
        
        # Force testing mode - always use extreme threshold to ensure filtering works
        force_test = os.environ.get('MEDUSA_FORCE_TEST', '').lower() in ('true', '1', 't', 'yes')
        if force_test:
            print("\n⚠️ TESTING MODE ENABLED: Forcing extreme threshold to demonstrate filtering")
            probability_threshold = 0.99  # Force high threshold for testing
        
        # Create Medusa decoder with our settings
        decoder = MedusaDecoder(
            model=self.base_model,  # Use base model directly, not self (to avoid recursion)
            tokenizer=tokenizer,
            medusa_heads=self.medusa_num_heads,
            probability_threshold=probability_threshold,
            debug=debug
        )
        
        # Extract generation parameters
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        temperature = kwargs.get('temperature', 0.0)
        top_p = kwargs.get('top_p', 0.9)
        
        # Use our decoder's generate method
        print("\n--- GENERATION STEP 1 ---")
        print("Testing generate_candidates...\n")
        
        # Use our decoder's generate method directly
        return decoder.generate(
            prompt_tokens=args[0] if args else kwargs.get('input_ids'),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
    
    @classmethod
    def from_pretrained(cls, model, medusa_num_heads=4, medusa_num_layers=1):
        """
        Create a MedusaModel from a pretrained model.
        
        Args:
            model: Pretrained model
            medusa_num_heads: Number of Medusa heads
            medusa_num_layers: Number of layers in each Medusa head
            
        Returns:
            MedusaModel instance
        """
        # Get model configuration
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Model does not have a config attribute")
            
        # Get hidden size and vocab size
        hidden_size = getattr(config, "hidden_size", None)
        vocab_size = getattr(config, "vocab_size", None)
        
        if hidden_size is None or vocab_size is None:
            raise ValueError("Model config must have hidden_size and vocab_size")
            
        # Create MedusaModel
        return cls(
            model,
            hidden_size,
            vocab_size,
            medusa_num_heads,
            medusa_num_layers
        )
        
    def save_pretrained(self, save_directory):
        """
        Save the MedusaModel to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(save_directory)
        
        # Save Medusa heads
        torch.save(
            self.medusa_heads.state_dict(),
            os.path.join(save_directory, "medusa_heads.pt")
        )
        
        # Save Medusa configuration
        with open(os.path.join(save_directory, "medusa_config.json"), "w") as f:
            json.dump({
                "medusa_num_heads": self.medusa_num_heads,
                "medusa_num_layers": self.medusa_num_layers,
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size
            }, f)
    
    @classmethod
    def load_medusa_heads(cls, model, save_directory):
        """
        Load Medusa heads from a directory.
        
        Args:
            model: MedusaModel instance
            save_directory: Directory where the model is saved
            
        Returns:
            Updated MedusaModel instance
        """
        # Load Medusa heads
        medusa_heads_path = os.path.join(save_directory, "medusa_heads.pt")
        if os.path.exists(medusa_heads_path):
            model.medusa_heads.load_state_dict(
                torch.load(medusa_heads_path, map_location=model.base_model.device)
            )
        
        return model 

# Default topk value for sparse tree
TOPK = 10  # This is a placeholder and usually sufficient

class MedusaDecoder:
    """
    Implements the Medusa decoding algorithm for faster LLM inference with probability threshold filtering.
    """
    
    def __init__(
        self, 
        model,
        tokenizer,
        medusa_heads: int = 4,
        tree_size: int = 5,
        max_candidates: int = 5,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        probability_threshold: float = 0.5,
        debug: bool = False,
        model_medusa_heads = None
    ):
        """
        Initialize the Medusa decoder.
        
        Args:
            model: The underlying model to use for decoding
            tokenizer: The tokenizer to use
            medusa_heads: Number of Medusa prediction heads
            tree_size: Maximum size of the tree to explore
            max_candidates: Maximum number of candidate sequences to consider
            posterior_threshold: Threshold for posterior validation
            posterior_alpha: Alpha parameter for posterior calculation (usually sqrt of threshold)
            probability_threshold: Threshold for filtering low-probability predictions from Medusa heads
            debug: Whether to print debug information
            model_medusa_heads: Optional. Directly provide model's Medusa heads instead of extracting them from the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.medusa_heads = medusa_heads
        self.tree_size = tree_size
        self.max_candidates = max_candidates
        self.posterior_threshold = posterior_threshold
        self.posterior_alpha = posterior_alpha
        
        # Check for environment variables
        # Get probability threshold from environment variable if available
        env_prob_threshold = os.environ.get('MEDUSA_PROBABILITY_THRESHOLD')
        if env_prob_threshold is not None:
            try:
                probability_threshold = float(env_prob_threshold)
                if debug:
                    print(f"Using probability threshold from environment: {probability_threshold}")
            except ValueError:
                pass  # Invalid float, stick with the provided value
        
        self.probability_threshold = probability_threshold
        
        # Check for debug environment variable
        env_debug = os.environ.get('MEDUSA_DEBUG', '').lower() in ('true', '1', 't', 'yes')
        self.debug = debug or env_debug
        
        # Check for force filtering mode (for testing)
        self.force_filtering = os.environ.get('MEDUSA_FORCE_FILTER', '').lower() in ('true', '1', 't', 'yes')
        if self.force_filtering and self.debug:
            print("\n⚠️ FORCE FILTERING MODE ENABLED - Will filter tokens below probability threshold")
            print(f"Probability threshold: {self.probability_threshold}")
        
        if self.debug:
            print(f"MedusaDecoder.__init__ called with probability_threshold={probability_threshold}")
            print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
        
        # Set default medusa choices based on the number of heads
        self.medusa_choices = self._get_default_medusa_choices(medusa_heads)
        
        # Initialize buffers
        self.medusa_buffers = None
        
        # Counter for tracking filtered tokens
        self._filtered_count = 0
        self._total_filtered_count = 0
        
        # Get the Medusa heads from the model if it's a MedusaModel
        self.medusa = None
        
        # First check if model_medusa_heads was directly provided
        if model_medusa_heads is not None:
            self.medusa = model_medusa_heads
            if self.debug:
                print(f"Using {len(self.medusa)} Medusa heads passed directly to decoder")
        # Otherwise try to find them in the model
        elif hasattr(model, 'medusa_heads'):
            self.medusa = model.medusa_heads
            if self.debug:
                print(f"Using {len(self.medusa)} Medusa heads from model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'medusa_heads'):
            self.medusa = model.base_model.medusa_heads
            if self.debug:
                print(f"Using {len(self.medusa)} Medusa heads from base model")
                
        # Ensure Medusa heads are in float32 to avoid dtype mismatches
        if self.medusa is not None:
            # Convert model parameters to float32 if they aren't already
            for head in self.medusa:
                for param in head.parameters():
                    if param.dtype != torch.float32:
                        if self.debug:
                            print(f"Converting Medusa head parameters from {param.dtype} to float32")
                        param.data = param.data.to(torch.float32)
    
    def _get_default_medusa_choices(self, num_heads):
        """Get default Medusa choices based on number of heads."""
        if num_heads == 1:
            return [[0]]
        elif num_heads == 2:
            return [[0], [0, 0]]
        elif num_heads == 4:
            return [[0], [0, 0], [0, 1], [0, 0, 0]]
        elif num_heads == 5:
            return [[0], [0, 0], [0, 1], [0, 0, 0], [0, 0, 1]]
        else:
            # Default fallback
            return [[i] for i in range(min(num_heads, 8))]
    
    def generate_medusa_buffers(self, medusa_choices, device="cuda"):
        """
        Generate buffers for the Medusa structure based on the provided choices.
        
        Args:
            medusa_choices: A nested list representing tree in the Medusa structure
            device: Device to which the tensors should be moved
            
        Returns:
            Dict containing buffers related to the Medusa structure
        """
        # Sort the medusa_choices based on their lengths and then their values
        sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
        medusa_len = len(sorted_medusa_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_medusa_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth
        
        # Create the attention mask for Medusa
        medusa_attn_mask = torch.eye(medusa_len, medusa_len)
        medusa_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                # retrieve ancestor position
                if len(cur_medusa_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_medusa_choice) - 1):
                    ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
                medusa_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        # Generate tree indices for the Medusa structure
        medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
        medusa_tree_indices[0] = 0
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
            start += depth_counts[i]

        # Generate position IDs for the Medusa structure
        medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        # Generate retrieval indices for Medusa structure verification
        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_medusa_choices)):
            cur_medusa_choice = sorted_medusa_choices[-i-1]
            retrieve_indice = []
            if cur_medusa_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_medusa_choice)):
                    retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                    retrieve_paths.append(cur_medusa_choice[:c+1])
            retrieve_indices_nest.append(retrieve_indice)
        
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [self._pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

        # Aggregate the generated buffers into a dictionary
        medusa_buffers = {
            "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": medusa_tree_indices,
            "medusa_position_ids": medusa_position_ids,
            "retrieve_indices": retrieve_indices,
        }
        
        # Move the tensors in the dictionary to the specified device
        medusa_buffers = {
            k: v.clone().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, device=device)
            for k, v in medusa_buffers.items()
        }
        return medusa_buffers
    
    def _generate_tree_indices(self):
        """
        Generate tree indices for Medusa decoding.
        
        Returns:
            Tensor of tree indices
        """
        # Initialize medusa buffers if not done yet
        if not hasattr(self, 'medusa_buffers') or self.medusa_buffers is None:
            # Determine device
            device = next(self.model.parameters()).device
            self.medusa_buffers = self.generate_medusa_buffers(self.medusa_choices, device=device)
            
        # Return the tree indices from medusa buffers
        return self.medusa_buffers["tree_indices"]
    
    def _generate_retrieve_indices(self):
        """
        Generate retrieval indices for Medusa decoding.
        
        Returns:
            Tensor of retrieval indices
        """
        # Initialize medusa buffers if not done yet
        if not hasattr(self, 'medusa_buffers') or self.medusa_buffers is None:
            # Determine device
            device = next(self.model.parameters()).device
            self.medusa_buffers = self.generate_medusa_buffers(self.medusa_choices, device=device)
            
        # Return the retrieval indices from medusa buffers
        return self.medusa_buffers["retrieve_indices"]
    
    def _pad_path(self, path, length, pad_value=-2):
        """Pad the given path list with a specific value up to a specified length."""
        return path + [pad_value] * (length - len(path))
    
    def generate_candidates(self, medusa_logits, logits, tree_indices, retrieve_indices, temperature=0.0, top_p=0.8):
        """
        Generate candidates with topk predictions from Medusa heads, filtering low-probability predictions.
        
        Args:
            medusa_logits: Logits from the Medusa heads
            logits: Logits from the base model
            tree_indices: Indices for the tree structure
            retrieve_indices: Indices for retrieving from the tree
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of candidates and tree candidates
        """
        if self.debug:
            print(f"\n🔍 DIAGNOSTIC: generate_candidates() called with probability_threshold={self.probability_threshold}")
            print(f"Medusa logits shape: {[logit.shape for logit in medusa_logits]}")
        
        # Track filtered tokens count for this method call
        filtered_count = 0
        
        with torch.no_grad():
            # Process base model prediction
            if temperature > 0:
                # Apply temperature and top-p sampling
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # Keep the top token
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                base_pred = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                base_pred = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                
                if self.debug:
                    # Get the base token for debugging
                    base_token = base_pred.item() if base_pred.numel() == 1 else base_pred[0].item() 
                    base_token_text = self.tokenizer.decode([base_token]) if hasattr(self.tokenizer, 'decode') else f"Token {base_token}"
                    print(f"Base model token: {base_token} ({base_token_text})")
            
            # Convert Medusa logits and tree indices to the right shapes
            medusa_logits = [logit[:, -1:, :] for logit in medusa_logits]
            n_medusa_tokens = len(medusa_logits)
            
            # Get predictions from each Medusa head
            medusa_preds = []
            if self.debug:
                print(f"Processing {n_medusa_tokens} Medusa heads with probability threshold {self.probability_threshold}")
            
            for i in range(n_medusa_tokens):
                # Get probabilities from logits
                medusa_probs = torch.softmax(medusa_logits[i], dim=-1)
                # Get the highest probability token
                max_prob_val, max_token_idx = torch.max(medusa_probs, dim=-1)
                
                # Get the token probability (this is the critical value for filtering)
                prob_val = max_prob_val.item() if max_prob_val.numel() == 1 else max_prob_val[0, 0].item()
                token_id = max_token_idx.item() if max_token_idx.numel() == 1 else max_token_idx[0, 0].item()
                
                # Get the token text for better readability
                token_text = self.tokenizer.decode([token_id]) if hasattr(self.tokenizer, 'decode') else f"Token {token_id}"
                
                # Check if we're in force filtering mode or if probability is below threshold
                force_filter = False
                if self.force_filtering:
                    # Apply filtering based on probability threshold
                    force_filter = prob_val < self.probability_threshold
                
                # Print debug information if enabled
                if self.debug:
                    print(f"Medusa Head {i+1}: Token '{token_text}' (ID: {token_id})")
                    print(f"Probability: {prob_val:.6f} vs threshold: {self.probability_threshold}")
                    if prob_val < self.probability_threshold or force_filter:
                        print(f"⚠️ BELOW THRESHOLD - Will be filtered")
                    else:
                        print(f"✅ ABOVE THRESHOLD - Will be kept")
                
                # APPLY PROBABILITY FILTERING
                if prob_val < self.probability_threshold or force_filter:
                    # Increment filtered count
                    filtered_count += 1
                    
                    if self.debug:
                        print(f"🔴 TOKEN FILTERED: Head {i+1} - '{token_text}' ({token_id})")
                        print(f"Probability: {prob_val:.6f} < threshold {self.probability_threshold}")
                    
                    # Use padding token instead of the low-probability prediction
                    default_token_id = 0  # Padding token
                    medusa_pred = torch.full_like(base_pred, default_token_id)
                else:
                    # Use the predicted token (probability >= threshold)
                    medusa_pred = max_token_idx.view_as(base_pred)  # Ensure same dims as base_pred
                    
                    if self.debug:
                        print(f"✅ TOKEN ACCEPTED: Head {i+1} - '{token_text}' ({token_id})")
                        print(f"Probability: {prob_val:.6f} >= threshold {self.probability_threshold}")
                
                # Add to medusa predictions
                medusa_preds.append(medusa_pred)
            
            # Update filtered tokens count
            self._filtered_count = filtered_count
            self._total_filtered_count += filtered_count
            
            # Print filtering summary if debugging
            if self.debug:
                print(f"\n==== TOKEN FILTERING SUMMARY =====")
                print(f"Tokens filtered this round: {filtered_count}")
                print(f"Total tokens filtered: {self._total_filtered_count}")
                if filtered_count > 0:
                    print(f"✅ Filtering is working - filtered {filtered_count} tokens below probability {self.probability_threshold}")
                else:
                    print(f"⚠️ No tokens filtered this round - all token probabilities were above {self.probability_threshold}")
                print(f"================================\n")
            
            # Generate tree candidates
            tree_candidates = [base_pred]  # base_pred is already [batch_size, 1]
            tree_candidates.extend(medusa_preds)  # All medusa_preds have same dim as base_pred
            
            # Concatenate along dim 0
            tree_candidates = torch.cat(tree_candidates, dim=0)
            
            # Generate candidates for verification
            candidates = []
            for i in range(retrieve_indices.shape[0]):
                candidate = []
                for j in range(1, retrieve_indices.shape[1]):
                    if retrieve_indices[i, j] >= 0:
                        idx = retrieve_indices[i, j]
                        if idx < tree_candidates.shape[0]:  # Safety check
                            candidate.append(tree_candidates[idx].item())
                candidates.append(candidate)
            
            if self.debug:
                print(f"Generated {len(candidates)} candidates with {filtered_count} tokens filtered")
                
            return candidates, tree_candidates
    
    def evaluate_posterior(self, logits, candidates, temperature=0.0, top_p=0.8):
        """
        Evaluate the posterior of the candidates to select the accepted candidate prefix.
        
        Args:
            logits: Logits from the model
            candidates: List of candidate token sequences
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of best candidate and accepted length
        """
        if not candidates:
            return [], 0
            
        # Filter out empty candidates
        valid_candidates = [c for c in candidates if c]
        if not valid_candidates:
            return [], 0
            
        # Compute posterior probabilities for each candidate
        best_candidate_idx = 0
        max_length = len(valid_candidates[0])
        
        # Simple implementation: use the first candidate
        best_candidate = valid_candidates[0]
        accept_length = len(best_candidate)
        
        return best_candidate, accept_length
    
    def update_inference_inputs(self, input_ids, candidates, best_candidate, accept_length):
        """
        Update the input_ids and prepare for the next round of inference.
        
        Args:
            input_ids: Current input token IDs
            candidates: List of candidate sequences
            best_candidate: The selected best candidate
            accept_length: Number of tokens to accept
            
        Returns:
            Updated input_ids, new_token count
        """
        # Update input_ids with the accepted tokens
        tokens_to_add = best_candidate[:accept_length]
        
        # Print information about the tokens being added
        if accept_length > 1 or self.debug:
            decoded_tokens = self.tokenizer.decode(tokens_to_add) if hasattr(self.tokenizer, 'decode') else f"Tokens: {tokens_to_add}"
            print(f"Medusa generated {accept_length} tokens in one step: {decoded_tokens}")
            
            if self._filtered_count > 0:
                print(f"Filtered {self._filtered_count} low-probability tokens using threshold {self.probability_threshold}")
        
        # Add the tokens to input_ids
        for token in tokens_to_add:
            input_ids = torch.cat([input_ids, torch.tensor([[token]], device=input_ids.device)], dim=1)
        
        return input_ids, len(tokens_to_add)
    
    def generate(
        self,
        prompt_tokens,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.8
    ):
        """
        Generate tokens using Medusa speculative parallel decoding with probability threshold filtering.
        
        Args:
            prompt_tokens: Input token IDs (can be tensor or list)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tensor of generated token IDs
        """
        if self.debug:
            print(f"\n 🚀🚀🚀 MEDUSA SPECULATIVE PARALLEL GENERATION 🚀🚀🚀")
            print(f"====================================================================")
            print(f"Active features:")
            print(f" - Multi-token parallel prediction: ENABLED")
            print(f" - Probability threshold filtering: {self.probability_threshold}")
            print(f" - Debug mode: VERBOSE")
            print(f" - Force filtering: {self.force_filtering}")
            print(f" - Medusa heads: {self.medusa_heads}")
            print(f"====================================================================")
        
        # Create a detailed debug log file to track parallel generation
        import os, time
        log_dir = os.path.expanduser("~/medusa_debug")
        os.makedirs(log_dir, exist_ok=True)
        debug_log = os.path.join(log_dir, f"medusa_debug_{int(time.time())}.log")
        
        if self.debug:
            with open(debug_log, "w") as f:
                f.write(f"MEDUSA PARALLEL GENERATION DEBUG LOG\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Probability threshold: {self.probability_threshold}\n")
                f.write(f"Model: {type(self.model).__name__}\n")
                f.write(f"Medusa heads: {self.medusa_heads}\n\n")
                f.write(f"=== GENERATION SETTINGS ===\n")
                f.write(f"max_new_tokens: {max_new_tokens}\n")
                f.write(f"temperature: {temperature}\n")
                f.write(f"top_p: {top_p}\n\n")
            print(f"\n📝 Detailed debug log will be saved to: {debug_log}")
        
        # Performance tracking
        start_time = time.time()
        total_iterations = 0
        total_tokens_generated = 0
        parallel_speedup_tokens = 0  # Tracks tokens generated in parallel (extra tokens)
        token_stats = []  # Track all token generation stats
        
        # Reset filtered token counters for this generation
        self._filtered_count = 0
        self._total_filtered_count = 0
        
        # Convert list or array to tensor if needed
        if not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.long)
        
        # Ensure prompt has batch dimension
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        
        # Move to the right device if needed
        device = None
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif next(self.model.parameters(), None) is not None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        prompt_tokens = prompt_tokens.to(device)
        input_ids = prompt_tokens
        
        if self.debug:
            print(f"\n🔄 STARTING MEDUSA PARALLEL GENERATION LOOP")
            print(f"Target: Generate up to {max_new_tokens} tokens using speculative decoding")
        
        # Generate tokens using Medusa speculative decoding
        generated_tokens = 0
        
        while generated_tokens < max_new_tokens:
            iteration_start = time.time()
            total_iterations += 1
            
            if self.debug:
                print(f"\n=== ITERATION {total_iterations} ===")
                print(f"Tokens generated so far: {generated_tokens}/{max_new_tokens}")
                if total_iterations == 1:
                    prompt_text = self.tokenizer.decode(input_ids[0]) if hasattr(self.tokenizer, 'decode') else "[Prompt tokens]"
                    print(f"\nPrompt: {prompt_text}\n")
                print(f"Running model forward pass...")
            
            # Run the model forward
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            # Get base model logits for next token prediction
            logits = outputs.logits
            
            # Check if we can use Medusa for parallel generation
            can_use_medusa = hasattr(self.model, 'medusa_heads') and self.model.medusa_heads is not None
            if not can_use_medusa and hasattr(self.model, 'base_model'):
                can_use_medusa = hasattr(self.model.base_model, 'medusa_heads') and self.model.base_model.medusa_heads is not None
            
            if can_use_medusa:
                if self.debug:
                    print(f"🐍 MEDUSA PARALLEL MODE: Generating multiple tokens in one step")
                
                # Get the last hidden states - handle different output formats
                # For CausalLMOutputWithPast (used by LlamaForCausalLM), we need to access hidden states differently
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # If hidden_states is a tuple containing hidden states from each layer, use the last layer
                    if isinstance(outputs.hidden_states, tuple):
                        last_hidden_state = outputs.hidden_states[-1]
                    else:
                        last_hidden_state = outputs.hidden_states
                else:
                    # For models that don't provide hidden states directly, we need to do a forward pass with output_hidden_states=True
                    if self.debug:
                        print("Model doesn't provide hidden states directly, running additional forward pass")
                    with torch.no_grad():
                        # Re-run with output_hidden_states=True
                        outputs = self.model(input_ids, output_hidden_states=True)
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            if isinstance(outputs.hidden_states, tuple):
                                last_hidden_state = outputs.hidden_states[-1]
                            else:
                                last_hidden_state = outputs.hidden_states
                        else:
                            if self.debug:
                                print("❌ Cannot get hidden states from model - Medusa requires access to hidden states")
                                print("Available attributes in output:", dir(outputs))
                            # Fallback to standard token-by-token generation
                            continue
                
                # Handle dtype compatibility issues by converting to float32 if needed
                if self.debug:
                    print(f"Hidden state dtype: {last_hidden_state.dtype}")
                
                # Convert hidden states to float32 to ensure compatibility with Medusa heads
                # This is needed because models often run in half precision but Medusa heads might expect full precision
                last_hidden_state = last_hidden_state.to(torch.float32)
                
                # Get Medusa logits from model's heads
                if hasattr(self.model, 'medusa_heads'):
                    medusa_logits = [head(last_hidden_state) for head in self.model.medusa_heads]
                    if self.debug:
                        print(f"Using {len(self.model.medusa_heads)} Medusa heads directly from model")
                elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'medusa_heads'):
                    medusa_logits = [head(last_hidden_state) for head in self.model.base_model.medusa_heads]
                    if self.debug:
                        print(f"Using {len(self.model.base_model.medusa_heads)} Medusa heads from base model")
                else:
                    # Should not happen due to the can_use_medusa check
                    if self.debug:
                        print("❌ No Medusa heads found despite can_use_medusa=True. This is a bug.")
                    continue
                
                # SPECULATIVE PARALLEL DECODING STEPS
                if self.debug:
                    print(f"\n📊 PARALLEL DECODING STEPS:")
                    print(f"1. Generate tree indices and retrieval indices")
                
                # Generate tree indices and retrieval indices for Medusa
                tree_indices = self._generate_tree_indices()
                retrieve_indices = self._generate_retrieve_indices()
                
                if self.debug:
                    print(f"2. Generate candidate token sequences with probability filtering")
                
                # STAGE 1: Generate candidate tokens using Medusa heads WITH THRESHOLD FILTERING
                candidates_start = time.time()
                candidates, tree_candidates = self.generate_candidates(
                    medusa_logits, logits, tree_indices, retrieve_indices,
                    temperature=temperature, top_p=top_p
                )
                candidates_time = time.time() - candidates_start
                
                if self.debug:
                    print(f"3. Evaluate candidate sequences and select best prefix")
                    if candidates:
                        print(f"   Generated {len(candidates)} candidate sequences")
                        for i, candidate in enumerate(candidates):
                            if i < 5:  # Only show the first 5 for brevity
                                tokens_text = self.tokenizer.decode(candidate) if hasattr(self.tokenizer, 'decode') else str(candidate)
                                print(f"   - Candidate {i+1}: {tokens_text} (length: {len(candidate)})")
                        if len(candidates) > 5:
                            print(f"   ... and {len(candidates)-5} more candidates")
                    else:
                        print(f"   ⚠️ No valid candidates generated")
                
                # STAGE 2: Select best candidate based on posterior probabilities
                posterior_start = time.time()
                best_candidate, accept_length = self.evaluate_posterior(
                    logits, candidates, temperature=temperature, top_p=top_p
                )
                posterior_time = time.time() - posterior_start
                
                if self.debug:
                    print(f"4. Update input with accepted tokens")
                    if accept_length > 0:
                        accepted_text = self.tokenizer.decode(best_candidate[:accept_length]) if hasattr(self.tokenizer, 'decode') else str(best_candidate[:accept_length])
                        print(f"   ✓ Accepted {accept_length} tokens: {accepted_text}")
                    else:
                        print(f"   ⚠️ No tokens accepted in this iteration")
                
                # STAGE 3: Update input with accepted tokens
                update_start = time.time()
                input_ids, new_tokens = self.update_inference_inputs(
                    input_ids, candidates, best_candidate, accept_length
                )
                update_time = time.time() - update_start
                
                # Track parallel speedup - tokens beyond the first one are "bonus" from parallelism
                if new_tokens > 1:
                    parallel_bonus = new_tokens - 1
                    parallel_speedup_tokens += parallel_bonus
                    if self.debug:
                        print(f"🚀 PARALLEL SPEEDUP: Generated {new_tokens} tokens in one step (+{parallel_bonus} bonus tokens)")
                
                # Update generated tokens count
                generated_tokens += new_tokens
                total_tokens_generated += new_tokens
                
                # Record detailed stats for this iteration
                if self.debug:
                    token_stats.append({
                        "iteration": total_iterations,
                        "tokens_generated": new_tokens,
                        "candidates_count": len(candidates),
                        "filtered_count": self._filtered_count,
                        "time_candidates": candidates_time,
                        "time_posterior": posterior_time,
                        "time_update": update_time,
                        "parallel_bonus": new_tokens - 1 if new_tokens > 1 else 0
                    })
                    
                    # Log to debug file
                    with open(debug_log, "a") as f:
                        f.write(f"\n=== ITERATION {total_iterations} ===\n")
                        f.write(f"Tokens generated: {new_tokens}\n")
                        f.write(f"Candidates count: {len(candidates)}\n")
                        f.write(f"Tokens filtered: {self._filtered_count}\n")
                        f.write(f"Parallel bonus: {new_tokens - 1 if new_tokens > 1 else 0}\n")
                        f.write(f"Processing times: Candidates={candidates_time:.4f}s, Posterior={posterior_time:.4f}s, Update={update_time:.4f}s\n")
                
                # If no tokens were accepted or we've generated enough tokens, we're done
                if new_tokens == 0 or generated_tokens >= max_new_tokens:
                    if self.debug:
                        print(f"{'No tokens accepted' if new_tokens == 0 else 'Reached max tokens'} - ending generation")
                    break
                
                # Continue to next iteration since we've already added tokens using Medusa
                continue
            
            # FALLBACK - Standard token-by-token prediction if Medusa is not available
            if self.debug:
                print(f"⚠️ FALLBACK MODE: Standard token-by-token generation (Medusa unavailable)")
            
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature if needed for more varied probabilities
            if temperature > 0:
                # Sampling with temperature
                next_token_logits = next_token_logits / temperature
                # Sample from the probability distribution
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                # Greedy selection (always pick highest probability token)
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
            # Add the new token to our sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens += 1
            total_tokens_generated += 1
            
            # Record standard generation stats
            if self.debug:
                token_id = next_token.item() if next_token.numel() == 1 else next_token[0].item()
                token_text = self.tokenizer.decode([token_id]) if hasattr(self.tokenizer, 'decode') else f"Token {token_id}"
                print(f"Generated 1 token: {token_text}")
                
                token_stats.append({
                    "iteration": total_iterations,
                    "tokens_generated": 1,
                    "candidates_count": 0,
                    "filtered_count": 0,
                    "parallel_bonus": 0
                })
                    
                # Check if we've generated enough tokens
                if generated_tokens >= max_new_tokens:
                    if self.debug:
                        print(f"Reached max tokens - ending generation")
                    break
                    
            # Record iteration time for this round
            iteration_time = time.time() - iteration_start
            if self.debug:
                print(f"\nIteration {total_iterations} completed in {iteration_time:.4f} seconds")
        
        # Print generation summary with detailed performance metrics
        total_time = time.time() - start_time
        tokens_per_second = total_tokens_generated / total_time if total_time > 0 else 0
        parallel_efficiency = (parallel_speedup_tokens / total_tokens_generated) * 100 if total_tokens_generated > 0 else 0
        
        if self.debug:
            print(f"\n🏁 GENERATION COMPLETE")
            print(f"====================================================================")
            print(f"Performance metrics:")
            print(f" - Total time: {total_time:.2f} seconds")
            print(f" - Total iterations: {total_iterations}")
            print(f" - Total tokens generated: {generated_tokens}")
            print(f" - Tokens per second: {tokens_per_second:.2f}")
            print(f" - Parallel speedup tokens: {parallel_speedup_tokens} (bonus tokens from parallel generation)")
            print(f" - Parallel efficiency: {parallel_efficiency:.2f}% (higher is better)")
            print(f" - Tokens filtered: {self._total_filtered_count}")
            print(f"====================================================================")
            
            # Log full generation stats to file
            with open(debug_log, "a") as f:
                f.write(f"\n=== GENERATION SUMMARY ===\n")
                f.write(f"Total time: {total_time:.2f} seconds\n")
                f.write(f"Total iterations: {total_iterations}\n")
                f.write(f"Total tokens generated: {generated_tokens}\n")
                f.write(f"Tokens per second: {tokens_per_second:.2f}\n")
                f.write(f"Parallel speedup tokens: {parallel_speedup_tokens}\n")
                f.write(f"Parallel efficiency: {parallel_efficiency:.2f}%\n")
                f.write(f"Tokens filtered: {self._total_filtered_count}\n\n")
                
                f.write(f"=== PER-ITERATION STATS ===\n")
                for i, stats in enumerate(token_stats):
                    if i < len(token_stats):
                        f.write(f"Iteration {i+1}:\n")
                        for k, v in stats.items():
                            if k != "iteration":
                                f.write(f"  {k}: {v}\n")
                        f.write("\n")
            
            # Print final result
            output_text = self.tokenizer.decode(input_ids[0]) if hasattr(self.tokenizer, 'decode') else "[Output tokens]"
            print(f"\n📝 Generated text:\n{output_text}")
            print(f"\n📋 Full debug log saved to: {debug_log}")
        
        return input_ids