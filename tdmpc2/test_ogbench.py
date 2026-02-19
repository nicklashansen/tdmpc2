#!/usr/bin/env python
"""
Test script to verify OGBench integration with TD-MPC2.

This script tests:
1. OGBench installation
2. Environment creation
3. Basic environment interaction
4. Observation and action space compatibility

Usage:
    python test_ogbench.py
    python test_ogbench.py --task ogb-humanoidmaze-large-navigate-v0
"""

import argparse
import sys
import traceback


def test_ogbench_installation():
    """Test if OGBench is installed."""
    print("=" * 60)
    print("Test 1: Checking OGBench installation...")
    print("=" * 60)
    
    try:
        import ogbench
        print("✓ OGBench is installed")
        if hasattr(ogbench, '__version__'):
            print(f"  Version: {ogbench.__version__}")
        return True
    except ImportError as e:
        print("✗ OGBench is not installed")
        print(f"  Error: {e}")
        print("\nTo install OGBench, run:")
        print("  pip install ogbench")
        print("\nOr if using a custom version:")
        print("  git clone <ogbench-repo-url>")
        print("  cd ogbench && pip install -e .")
        return False


def test_gymnasium():
    """Test if gymnasium is installed."""
    print("\n" + "=" * 60)
    print("Test 2: Checking Gymnasium installation...")
    print("=" * 60)
    
    try:
        import gymnasium as gym
        print("✓ Gymnasium is installed")
        print(f"  Version: {gym.__version__}")
        return True
    except ImportError as e:
        print("✗ Gymnasium is not installed")
        print(f"  Error: {e}")
        return False


def test_env_creation(task='ogb-humanoidmaze-large-navigate-v0'):
    """Test environment creation."""
    print("\n" + "=" * 60)
    print(f"Test 3: Creating OGBench environment ({task})...")
    print("=" * 60)
    
    try:
        # Import the config parser
        from omegaconf import OmegaConf
        
        # Create minimal config
        cfg = OmegaConf.create({
            'task': task,
            'obs': 'state',
            'seed': 42,
            'multitask': False
        })
        
        # Import and create environment
        from envs import make_env
        env = make_env(cfg)
        
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Max episode steps: {env.max_episode_steps}")
        
        return env
    except Exception as e:
        print(f"✗ Failed to create environment")
        print(f"  Error: {e}")
        traceback.print_exc()
        return None


def test_env_interaction(env):
    """Test basic environment interaction."""
    print("\n" + "=" * 60)
    print("Test 4: Testing environment interaction...")
    print("=" * 60)
    
    if env is None:
        print("✗ Cannot test interaction - environment not created")
        return False
    
    try:
        # Reset environment
        obs = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # Take a random action
        action = env.action_space.sample()
        print(f"✓ Sampled random action")
        print(f"  Action shape: {action.shape if hasattr(action, 'shape') else type(action)}")
        
        # Step environment
        obs, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info keys: {info.keys() if isinstance(info, dict) else 'N/A'}")
        
        # Test multiple steps
        print("\nRunning 10 random steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                print(f"  Episode ended at step {i+1}")
                obs = env.reset()
        
        print("✓ All 10 steps completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Environment interaction failed")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_tdmpc2_compatibility(env):
    """Test compatibility with TD-MPC2."""
    print("\n" + "=" * 60)
    print("Test 5: Testing TD-MPC2 compatibility...")
    print("=" * 60)
    
    if env is None:
        print("✗ Cannot test compatibility - environment not created")
        return False
    
    try:
        import torch
        
        # Check if observations are torch tensors
        obs = env.reset()
        is_tensor = torch.is_tensor(obs)
        print(f"  Observation is torch tensor: {is_tensor}")
        
        if is_tensor:
            print(f"  Observation dtype: {obs.dtype}")
            print(f"  Observation device: {obs.device}")
        
        # Check action space
        action = env.action_space.sample()
        print(f"  Action space type: {type(env.action_space)}")
        print(f"  Action dimension: {env.action_space.shape[0]}")
        
        # Check episode length
        print(f"  Max episode steps: {env.max_episode_steps}")
        
        print("✓ Environment appears compatible with TD-MPC2")
        return True
        
    except Exception as e:
        print(f"✗ Compatibility check failed")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test OGBench integration with TD-MPC2')
    parser.add_argument('--task', type=str, default='ogb-humanoidmaze-large-navigate-v0',
                      help='OGBench task to test (default: ogb-humanoidmaze-large-navigate-v0)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("OGBench Integration Test Suite")
    print("=" * 60)
    
    # Run tests
    results = []
    env = None
    
    try:
        # Test 1: OGBench installation
        results.append(("OGBench Installation", test_ogbench_installation()))
        
        # Test 2: Gymnasium installation
        results.append(("Gymnasium Installation", test_gymnasium()))
        
        # Only continue if basic dependencies are installed
        if not all(r[1] for r in results):
            print("\n" + "=" * 60)
            print("SUMMARY: Basic dependencies missing")
            print("=" * 60)
            print("Please install missing dependencies before continuing.")
            sys.exit(1)
        
        # Test 3: Environment creation
        env = test_env_creation(args.task)
        results.append(("Environment Creation", env is not None))
        
        # Test 4: Environment interaction
        if env is not None:
            results.append(("Environment Interaction", test_env_interaction(env)))
            
            # Test 5: TD-MPC2 compatibility
            results.append(("TD-MPC2 Compatibility", test_tdmpc2_compatibility(env)))
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        for test_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        all_passed = all(r[1] for r in results)
        
        print("\n" + "=" * 60)
        if all_passed:
            print("SUCCESS: All tests passed! ✓")
            print("=" * 60)
            print("\nYou can now train TD-MPC2 on OGBench tasks:")
            print(f"  python train.py task={args.task} model_size=5 steps=1000000")
            sys.exit(0)
        else:
            print("FAILURE: Some tests failed! ✗")
            print("=" * 60)
            print("\nPlease fix the issues above before training.")
            sys.exit(1)
    
    finally:
        # Clean up environment to avoid EGL errors
        if env is not None:
            try:
                env.close()
            except:
                pass


if __name__ == '__main__':
    main()
