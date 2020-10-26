#!/bin/bash

# These are basic tests that are supposed just to check if the scripts run. Not
# really if they learn something.

set -ex
mkdir -p test_outputs

# EDIT DISTANCE GENERATION ===================================================

./train_transliteration_generation.py --model-type embeddings --hidden-size 16 --nll-loss 1.0 --sampled-em-loss 0.0 data/test_generation --distortion-loss 0.0 --final-state-loss 0.0 --batch-size 20 --contrastive-loss 0.0 --delay-update 2 --epochs 2 --validation-frequency 5 --em-loss 1.0 --log-directory test_outputs

./train_transliteration_generation.py --model-type rnn --hidden-size 16 --nll-loss 1.0 --sampled-em-loss 0.0 data/test_generation --distortion-loss 0.0 --final-state-loss 0.0 --batch-size 20 --contrastive-loss 0.0 --delay-update 2 --epochs 2 --validation-frequency 5 --em-loss 1.0 --no-enc-dec-att --log-directory test_outputs

./train_transliteration_generation.py --model-type transformer --hidden-size 16 --nll-loss 1.0 --sampled-em-loss 0.0 data/test_generation --distortion-loss 0.0 --final-state-loss 0.0 --batch-size 20 --contrastive-loss 0.0 --delay-update 2 --epochs 2 --validation-frequency 5 --em-loss 1.0 --no-enc-dec-att --log-directory test_outputs

./train_transliteration_generation.py --model-type rnn --hidden-size 16 --nll-loss 1.0 --sampled-em-loss 0.0 data/test_generation --distortion-loss 0.0 --final-state-loss 0.0 --batch-size 20 --contrastive-loss 0.0 --delay-update 2 --epochs 2 --validation-frequency 5 --em-loss 1.0 --log-directory test_outputs

./train_transliteration_generation.py --model-type transformer --hidden-size 16 --nll-loss 1.0 --sampled-em-loss 0.0 data/test_generation --distortion-loss 0.0 --final-state-loss 0.0 --batch-size 20 --contrastive-loss 0.0 --delay-update 2 --epochs 2 --validation-frequency 5 --em-loss 1.0 --log-directory test_outputs

# S2S MODELS =================================================================

./train_transliteration_s2s.py --model-type transformer --hidden-size 32 data/test_generation --batch-size 20 --epochs 2 --validation-frequency 5 --log-directory test_outputs

./train_transliteration_s2s.py --model-type rnn --hidden-size 32 data/test_generation --batch-size 20 --epochs 2 --validation-frequency 5 --log-directory test_outputs

# EDIT DISTANCE CLASSIFICATION ===============================================

./train_transliteration_classification.py --model-type embeddings --hidden-size 16 data/test_classification --interpretation-loss 0.0 --batch-size 20  --delay-update 2 --epochs 2 --validation-frequency 5 --log-directory test_outputs

./train_transliteration_classification.py --model-type rnn --hidden-size 16 data/test_classification --interpretation-loss 0.0 --batch-size 20  --delay-update 2 --epochs 2 --validation-frequency 5 --log-directory test_outputs

./train_transliteration_classification.py --model-type transformer --hidden-size 16 data/test_classification --interpretation-loss 0.0 --batch-size 20  --delay-update 2 --epochs 2 --validation-frequency 5 --log-directory test_outputs

# TRANSFORMER CLASSIFIER =====================================================

./train_baseline_cognates_classification.py --hidden-size 16 data/test_classification --batch-size 20  --delay-update 2 --epochs 2 --validation-frequency 5 --log-directory test_outputs
