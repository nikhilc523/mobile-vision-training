"""
LSTM Training Module for Fall Detection

This module provides training utilities for the LSTM-based fall detection model.
"""

from .lstm_train import train_lstm, build_model, evaluate_model

__all__ = ['train_lstm', 'build_model', 'evaluate_model']

