#!/usr/bin/env python3
"""
Test runner script for diabetic retinopathy project
"""
import sys
import subprocess
import argparse
import os


def run_tests(test_type='all', coverage=True, verbose=True):
    """Run tests with specified parameters"""
    
    # Base pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            '--cov=ml',
            '--cov=backend',
            '--cov-report=html:htmlcov',
            '--cov-report=term-missing'
        ])
    
    # Add verbosity
    if verbose:
        cmd.append('-v')
    
    # Select test type
    if test_type == 'unit':
        cmd.extend(['-m', 'unit'])
    elif test_type == 'integration':
        cmd.extend(['-m', 'integration'])
    elif test_type == 'fast':
        cmd.extend(['-m', 'not slow'])
    elif test_type == 'slow':
        cmd.extend(['-m', 'slow'])
    elif test_type != 'all':
        # Specific test file or directory
        cmd.append(test_type)
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run tests
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run tests for diabetic retinopathy project')
    
    parser.add_argument(
        '--type', 
        choices=['all', 'unit', 'integration', 'fast', 'slow'],
        default='all',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--no-coverage',
        action='store_true',
        help='Skip coverage reporting'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Run tests quietly'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Run specific test file'
    )
    
    args = parser.parse_args()
    
    # Determine test type
    test_type = args.file if args.file else args.type
    
    # Run tests
    exit_code = run_tests(
        test_type=test_type,
        coverage=not args.no_coverage,
        verbose=not args.quiet
    )
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
        if not args.no_coverage:
            print("📊 Coverage report generated in 'htmlcov/' directory")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()