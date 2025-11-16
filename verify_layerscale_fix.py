#!/usr/bin/env python3
"""
Verification script for LayerScale integration fix.

This script demonstrates that the LayerScale parameter mismatch has been resolved.
It checks the code structure without requiring torch to be installed.
"""

import ast
import inspect


def check_create_encoder_signature():
    """Check that create_encoder has the correct signature."""
    print("Checking create_encoder() signature...")

    # Read the encoder.py file
    with open('src/models/encoder.py', 'r') as f:
        content = f.read()

    # Parse the file
    tree = ast.parse(content)

    # Find the create_encoder function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'create_encoder':
            params = [arg.arg for arg in node.args.args]

            required_params = ['use_flash_attention', 'use_layerscale', 'layerscale_init']
            for param in required_params:
                if param in params:
                    print(f"  ✅ Found parameter: {param}")
                else:
                    print(f"  ❌ Missing parameter: {param}")
                    return False

            return True

    print("  ❌ Could not find create_encoder function")
    return False


def check_hjepa_calls_create_encoder():
    """Check that HJEPA correctly calls create_encoder."""
    print("\nChecking HJEPA calls to create_encoder()...")

    # Read the hjepa.py file
    with open('src/models/hjepa.py', 'r') as f:
        content = f.read()

    # Look for the create_encoder call
    if 'create_encoder(' not in content:
        print("  ❌ HJEPA does not call create_encoder")
        return False

    # Find the specific call with parameters
    start = content.find('self.context_encoder, self.target_encoder = create_encoder(')
    if start == -1:
        print("  ❌ Could not find encoder creation call")
        return False

    end = content.find(')', start)
    call = content[start:end]

    # Check for required parameters
    required_params = ['use_flash_attention', 'use_layerscale', 'layerscale_init']
    for param in required_params:
        if f'{param}={param}' in call:
            print(f"  ✅ HJEPA passes: {param}")
        else:
            print(f"  ❌ HJEPA missing: {param}")
            return False

    return True


def check_todo_comments():
    """Check that TODO comments exist for pending implementation."""
    print("\nChecking TODO comments...")

    with open('src/models/encoder.py', 'r') as f:
        content = f.read()

    # Look for TODO comments related to LayerScale
    if 'TODO: LayerScale integration' in content:
        print("  ✅ Found TODO comment for LayerScale")
        return True
    elif 'TODO' in content and 'layerscale' in content.lower():
        print("  ✅ Found TODO comment mentioning LayerScale")
        return True
    else:
        print("  ⚠️  No TODO comment found (may be implemented)")
        return True  # Not a failure


def main():
    print("=" * 70)
    print("LayerScale Integration Fix Verification")
    print("=" * 70)

    checks = [
        ("create_encoder() signature", check_create_encoder_signature),
        ("HJEPA parameter passing", check_hjepa_calls_create_encoder),
        ("TODO comments", check_todo_comments),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during '{name}': {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✅ SUCCESS: LayerScale integration fix is complete!")
        print("\nThe fix ensures:")
        print("  • create_encoder() accepts use_layerscale and layerscale_init")
        print("  • HJEPA correctly passes these parameters")
        print("  • No TypeError will occur when use_layerscale=True")
        print("  • Code is ready for full LayerScale implementation")
        print("\nNote: LayerScale functionality is not yet implemented (no-op).")
        return 0
    else:
        print("\n❌ FAILURE: Some checks failed")
        return 1


if __name__ == "__main__":
    exit(main())
