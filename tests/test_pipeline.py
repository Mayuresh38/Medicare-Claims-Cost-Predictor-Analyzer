import os
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import json

from src.data_loader import load_or_create_dataset
from src.preprocessing import preprocess_beneficiaries, preprocess_claims, build_longitudinal_sequences
from src.features import build_tabular_features

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        # Build a small local mock dataset for tests
        self.b_df, self.i_df = load_or_create_dataset(use_mock=True, num_bene=50, num_claims=100)

    def test_data_loader(self):
        self.assertGreater(len(self.b_df), 0)
        self.assertGreater(len(self.i_df), 0)
        self.assertIn("DESYNPUF_ID", self.b_df.columns)
        self.assertIn("DESYNPUF_ID", self.i_df.columns)

    def test_preprocessing(self):
        b_proc = preprocess_beneficiaries(self.b_df)
        i_proc = preprocess_claims(self.i_df)
        
        self.assertIn("AGE", b_proc.columns)
        self.assertIn("IS_FEMALE", b_proc.columns)
        self.assertIn("TOTAL_ANNUAL_COST", b_proc.columns)
        self.assertIn("CLAIM_DURATION", i_proc.columns)
        
    def test_longitudinal_sequences(self):
        max_seq_len = 5
        seqs = build_longitudinal_sequences(self.b_df, self.i_df, max_seq_len=max_seq_len)
        # Should have shape (num_benes, max_seq_len, 4)
        self.assertEqual(seqs.shape, (50, max_seq_len, 4))
        
    def test_tabular_features(self):
        X, y, cols = build_tabular_features(self.b_df, self.i_df)
        self.assertEqual(len(X), 50)
        self.assertEqual(len(y), 50)
        self.assertIn("AGE", X.columns)
        self.assertIn("TOTAL_CLAIM_PMT", X.columns)

    def test_saved_models_exist(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "scaler.pkl")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "features.json")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "best_model_info.json")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "model_comparison.csv")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "ann.keras")))

if __name__ == "__main__":
    unittest.main()
