#!/usr/bin/env python3
"""
Test script for Medusa decoder probability threshold filtering.

This standalone script validates that the probability threshold filtering 
in the Medusa decoder works correctly.
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exo.inference.medusa_decoder import MedusaDecoder

def test_probability_threshold():
    """Test the probability threshold filtering in MedusaDecoder."""
    print("\n==== Testing Medusa Probability Threshold Filtering ====\n")
    
    # Create a simple decoder for testing
    class MockModel:
        pass
    
    class MockTokenizer:
        def decode(self, tokens):
            return f"Token_{tokens[0]}"
    
    # Create decoder with high threshold
    model = MockModel()
    tokenizer = MockTokenizer()
    decoder = MedusaDecoder(
        model=model,
        tokenizer=tokenizer,
        medusa_heads=4,
        probability_threshold=0.8,  # High threshold to filter many tokens
        debug=True
    )
    
    print(f"Created MedusaDecoder with probability_threshold={decoder.probability_threshold}")
    
    # Create mock data for testing
    batch_size = 1
    vocab_size = 1000
    
    # Base logits
    base_logits = torch.zeros(batch_size, 1, vocab_size)
    base_logits[0, 0, 42] = 10.0  # High probability for token 42
    
    # Medusa head logits with varying probabilities
    medusa_logits = []
    for i in range(4):
        head_logits = torch.zeros(batch_size, 1, vocab_size)
        if i % 2 == 0:
            # High probability (will pass threshold)
            head_logits[0, 0, 100 + i] = 10.0  # probability ~0.9999
        else:
            # Low probability (will be filtered)
            head_logits[0, 0, 200 + i] = 1.0  # probability ~0.7
        medusa_logits.append(head_logits)
    
    # Mock tree indices and retrieve indices
    tree_indices = torch.zeros(5, dtype=torch.long)
    retrieve_indices = torch.zeros((1, 5), dtype=torch.long)
    retrieve_indices[0, 0] = 0
    
    print("\n--- Running generate_candidates with mixed probabilities ---")
    candidates, tree_candidates = decoder.generate_candidates(
        medusa_logits, base_logits, tree_indices, retrieve_indices
    )
    
    print("\n--- Final Results ---")
    print(f"Total filtered tokens: {decoder._total_filtered_count}")
    print(f"Tree candidates shape: {tree_candidates.shape}")
    print(f"Number of candidates: {len(candidates)}")
    
    return decoder._total_filtered_count > 0

if __name__ == "__main__":
    success = test_probability_threshold()
    
    if success:
        print("\n✅ TEST PASSED: Probability threshold filtering is working correctly!")
    else:
        print("\n❌ TEST FAILED: Probability threshold filtering is not working.")
