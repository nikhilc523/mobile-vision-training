"""
Convert trained Keras model to TensorFlow Lite format for mobile deployment.

This script:
1. Loads the best trained model (lstm_raw30_balanced_hnm_best.h5)
2. Converts to TensorFlow Lite format (.tflite)
3. Optimizes for mobile (quantization options)
4. Tests the converted model with sample input
5. Compares TFLite vs Keras predictions
6. Saves conversion report

Usage:
    python -m ml.export.convert_to_tflite
    python -m ml.export.convert_to_tflite --quantize  # For smaller model
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_keras_model(model_path):
    """Load the trained Keras model."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading Keras Model")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"\nModel Summary:")
    model.summary()
    
    # Get model info
    input_shape = model.input_shape
    output_shape = model.output_shape
    total_params = model.count_params()
    
    print(f"\n📊 Model Information:")
    print(f"   Input shape:  {input_shape}")
    print(f"   Output shape: {output_shape}")
    print(f"   Total params: {total_params:,}")
    
    return model, input_shape, output_shape


def convert_to_tflite(model, output_path, quantize=False):
    """Convert Keras model to TensorFlow Lite format."""
    print(f"\n{'='*60}")
    print("STEP 2: Converting to TensorFlow Lite")
    print(f"{'='*60}")

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # IMPORTANT: BiLSTM models need SELECT_TF_OPS for dynamic operations
    print("🔧 Configuring converter for BiLSTM model...")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Enable select TensorFlow ops (needed for LSTM)
    ]
    converter._experimental_lower_tensor_list_ops = False  # Don't lower tensor list ops

    if quantize:
        print("🔧 Applying dynamic range quantization (smaller model, faster inference)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        print("🔧 Converting without quantization (full precision)...")

    # Convert the model
    print("Converting model...")
    print("⚠️  Note: BiLSTM models will include some TensorFlow ops (not pure TFLite)")
    tflite_model = converter.convert()
    
    # Save the model
    print(f"Saving TFLite model to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file size
    file_size_kb = os.path.getsize(output_path) / 1024
    file_size_mb = file_size_kb / 1024
    
    print(f"✅ TFLite model saved successfully!")
    print(f"   File size: {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")
    
    return tflite_model, file_size_kb


def test_tflite_model(tflite_model_path, test_input):
    """Test the TFLite model with sample input."""
    print(f"\n{'='*60}")
    print("STEP 3: Testing TFLite Model")
    print(f"{'='*60}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📋 TFLite Model Details:")
    print(f"   Input details:")
    print(f"      Index: {input_details[0]['index']}")
    print(f"      Shape: {input_details[0]['shape']}")
    print(f"      Type:  {input_details[0]['dtype']}")
    print(f"   Output details:")
    print(f"      Index: {output_details[0]['index']}")
    print(f"      Shape: {output_details[0]['shape']}")
    print(f"      Type:  {output_details[0]['dtype']}")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], test_input)
    
    # Run inference
    print(f"\n🚀 Running inference...")
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Inference complete!")
    print(f"   Output shape: {output.shape}")
    print(f"   Output value: {output[0][0]:.6f}")
    
    return output, interpreter


def compare_models(keras_model, tflite_interpreter, test_input):
    """Compare Keras and TFLite model predictions."""
    print(f"\n{'='*60}")
    print("STEP 4: Comparing Keras vs TFLite")
    print(f"{'='*60}")
    
    # Keras prediction
    print("Running Keras model...")
    keras_output = keras_model.predict(test_input, verbose=0)
    keras_prob = keras_output[0][0]
    
    # TFLite prediction
    print("Running TFLite model...")
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    
    tflite_interpreter.set_tensor(input_details[0]['index'], test_input)
    tflite_interpreter.invoke()
    tflite_output = tflite_interpreter.get_tensor(output_details[0]['index'])
    tflite_prob = tflite_output[0][0]
    
    # Compare
    diff = abs(keras_prob - tflite_prob)
    diff_percent = (diff / keras_prob) * 100 if keras_prob > 0 else 0
    
    print(f"\n📊 Comparison Results:")
    print(f"   Keras prediction:  {keras_prob:.8f}")
    print(f"   TFLite prediction: {tflite_prob:.8f}")
    print(f"   Difference:        {diff:.8f} ({diff_percent:.4f}%)")
    
    if diff < 1e-5:
        print(f"   ✅ Models match perfectly!")
    elif diff < 1e-3:
        print(f"   ✅ Models match very closely (acceptable difference)")
    else:
        print(f"   ⚠️  Models have noticeable difference (may need investigation)")
    
    return keras_prob, tflite_prob, diff


def create_test_inputs():
    """Create test inputs for model testing."""
    print(f"\n{'='*60}")
    print("Creating Test Inputs")
    print(f"{'='*60}")
    
    # Test input shape: (batch_size, 30, 34)
    # 30 frames, 34 features (17 keypoints × 2 coordinates)
    
    # Test 1: Random normal input (simulates normal activity)
    test_normal = np.random.randn(1, 30, 34).astype(np.float32)
    test_normal = np.clip(test_normal * 0.1 + 0.5, 0, 1)  # Normalize to [0, 1]
    
    # Test 2: Simulated fall pattern
    # Fall characteristics: rapid descent, orientation change, stillness
    test_fall = np.random.randn(1, 30, 34).astype(np.float32)
    test_fall = np.clip(test_fall * 0.1 + 0.5, 0, 1)
    
    # Simulate rapid descent (y-coordinates decreasing)
    for t in range(10, 20):  # Frames 10-20: falling
        test_fall[0, t, ::2] -= 0.05 * (t - 10)  # y-coordinates decrease
    
    # Simulate stillness after fall (frames 20-30)
    for t in range(20, 30):
        test_fall[0, t, :] = test_fall[0, 19, :]  # Same position
    
    test_fall = np.clip(test_fall, 0, 1)
    
    # Test 3: All zeros (edge case)
    test_zeros = np.zeros((1, 30, 34), dtype=np.float32)
    
    print(f"✅ Created 3 test inputs:")
    print(f"   1. Normal activity (random)")
    print(f"   2. Simulated fall pattern")
    print(f"   3. All zeros (edge case)")
    
    return {
        'normal': test_normal,
        'fall': test_fall,
        'zeros': test_zeros
    }


def save_conversion_report(output_dir, results):
    """Save conversion report to file."""
    report_path = os.path.join(output_dir, 'tflite_conversion_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TensorFlow Lite Conversion Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"  Input shape:  {results['input_shape']}\n")
        f.write(f"  Output shape: {results['output_shape']}\n")
        f.write(f"  Total params: {results['total_params']:,}\n\n")
        
        f.write("Conversion Results:\n")
        f.write(f"  Keras model size:  {results['keras_size_kb']:.2f} KB\n")
        f.write(f"  TFLite model size: {results['tflite_size_kb']:.2f} KB\n")
        f.write(f"  Size reduction:    {results['size_reduction']:.1f}%\n\n")
        
        f.write("Test Results:\n")
        for test_name, test_result in results['test_results'].items():
            f.write(f"\n  Test: {test_name}\n")
            f.write(f"    Keras prediction:  {test_result['keras']:.8f}\n")
            f.write(f"    TFLite prediction: {test_result['tflite']:.8f}\n")
            f.write(f"    Difference:        {test_result['diff']:.8f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Conversion Status: ✅ SUCCESS\n")
        f.write("="*60 + "\n")
    
    print(f"\n📄 Conversion report saved to: {report_path}")


def main():
    """Main conversion function."""
    print("\n" + "="*60)
    print("TensorFlow Lite Model Conversion")
    print("="*60)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'ml' / 'training' / 'checkpoints' / 'lstm_raw30_balanced_hnm_best.h5'
    output_dir = project_root / 'ml' / 'export'
    output_dir.mkdir(exist_ok=True)
    
    tflite_path = output_dir / 'fall_detection_model.tflite'
    tflite_quant_path = output_dir / 'fall_detection_model_quantized.tflite'
    
    # Step 1: Load Keras model
    keras_model, input_shape, output_shape = load_keras_model(str(model_path))
    keras_size_kb = os.path.getsize(model_path) / 1024
    
    # Step 2: Convert to TFLite (full precision)
    tflite_model, tflite_size_kb = convert_to_tflite(keras_model, str(tflite_path), quantize=False)
    
    # Step 2b: Convert to TFLite (quantized)
    print(f"\n{'='*60}")
    print("STEP 2b: Converting to TensorFlow Lite (Quantized)")
    print(f"{'='*60}")
    tflite_quant_model, tflite_quant_size_kb = convert_to_tflite(keras_model, str(tflite_quant_path), quantize=True)
    
    # Step 3: Create test inputs
    test_inputs = create_test_inputs()
    
    # Step 4: Test TFLite model
    tflite_output, tflite_interpreter = test_tflite_model(str(tflite_path), test_inputs['normal'])
    
    # Step 5: Compare models on all test inputs
    test_results = {}
    
    for test_name, test_input in test_inputs.items():
        print(f"\n{'='*60}")
        print(f"Testing: {test_name.upper()}")
        print(f"{'='*60}")
        
        keras_prob, tflite_prob, diff = compare_models(keras_model, tflite_interpreter, test_input)
        
        test_results[test_name] = {
            'keras': keras_prob,
            'tflite': tflite_prob,
            'diff': diff
        }
        
        # Interpret result
        if keras_prob > 0.85:
            print(f"   🚨 FALL DETECTED (probability: {keras_prob:.2%})")
        else:
            print(f"   ✅ NO FALL (probability: {keras_prob:.2%})")
    
    # Step 6: Save conversion report
    results = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'total_params': keras_model.count_params(),
        'keras_size_kb': keras_size_kb,
        'tflite_size_kb': tflite_size_kb,
        'size_reduction': ((keras_size_kb - tflite_size_kb) / keras_size_kb) * 100,
        'test_results': test_results
    }
    
    save_conversion_report(str(output_dir), results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"\n✅ Models saved:")
    print(f"   Full precision:  {tflite_path}")
    print(f"                    Size: {tflite_size_kb:.2f} KB")
    print(f"   Quantized:       {tflite_quant_path}")
    print(f"                    Size: {tflite_quant_size_kb:.2f} KB")
    print(f"\n📊 Size comparison:")
    print(f"   Keras (.h5):     {keras_size_kb:.2f} KB")
    print(f"   TFLite (full):   {tflite_size_kb:.2f} KB ({results['size_reduction']:.1f}% reduction)")
    print(f"   TFLite (quant):  {tflite_quant_size_kb:.2f} KB ({((keras_size_kb - tflite_quant_size_kb) / keras_size_kb) * 100:.1f}% reduction)")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Use 'fall_detection_model.tflite' for Android Studio")
    print(f"   2. Input shape: (1, 30, 34) - batch_size=1, 30 frames, 34 features")
    print(f"   3. Output shape: (1, 1) - probability [0, 1]")
    print(f"   4. Threshold: 0.85 for fall detection")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()

