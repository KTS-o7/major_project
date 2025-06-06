#!/usr/bin/env python3
"""
Code Coverage Report Generator for CUDA Tests

This script generates comprehensive code coverage reports for the CUDA module tests.
It includes both terminal output and HTML reports with detailed analysis.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_coverage_tools():
    """Install pytest-cov if not already installed."""
    print("📦 Installing coverage tools...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov", "coverage"], 
                      check=True, capture_output=True)
        print("✅ Coverage tools installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install coverage tools: {e}")
        sys.exit(1)

def run_coverage_tests():
    """Run tests with coverage analysis."""
    print("\n🧪 Running CUDA tests with coverage analysis...")
    
    # Define the source and test directories
    source_dir = "logic/cuda"
    test_dir = "logic/cuda/tests"
    
    # Coverage command with comprehensive options
    coverage_cmd = [
        sys.executable, "-m", "pytest",
        test_dir,
        f"--cov={source_dir}",
        "--cov-report=term-missing",  # Terminal report with missing lines
        "--cov-report=html:coverage_html",  # HTML report
        "--cov-report=xml:coverage.xml",  # XML report for CI/CD
        "--cov-branch",  # Include branch coverage
        "--cov-fail-under=80",  # Fail if coverage below 80%
        "-v",  # Verbose output
        "--tb=short"  # Short traceback format
    ]
    
    try:
        result = subprocess.run(coverage_cmd, check=False, capture_output=True, text=True)
        print("📊 Test Results:")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings/Errors:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run coverage tests: {e}")
        return False

def generate_detailed_coverage_report():
    """Generate additional detailed coverage reports."""
    print("\n📈 Generating detailed coverage reports...")
    
    # Generate coverage report by file
    try:
        # Coverage by file
        subprocess.run([
            sys.executable, "-m", "coverage", "report", 
            "--show-missing", "--sort=cover"
        ], check=True)
        
        # Generate annotated source files
        subprocess.run([
            sys.executable, "-m", "coverage", "annotate"
        ], check=True)
        
        print("✅ Detailed reports generated successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate detailed reports: {e}")

def analyze_coverage_results():
    """Analyze and provide insights on coverage results."""
    print("\n🔍 Coverage Analysis Summary:")
    
    # Check if HTML report was generated
    html_report_path = Path("coverage_html/index.html")
    if html_report_path.exists():
        print(f"📄 HTML Report: {html_report_path.absolute()}")
        print("   Open this file in a browser for interactive coverage analysis")
    
    # Check for XML report
    xml_report_path = Path("coverage.xml")
    if xml_report_path.exists():
        print(f"📄 XML Report: {xml_report_path.absolute()}")
        print("   Use this for CI/CD integration")
    
    # Provide recommendations
    print("\n💡 Coverage Improvement Recommendations:")
    print("   1. Focus on files with < 80% coverage")
    print("   2. Add tests for uncovered branches and edge cases")
    print("   3. Consider integration tests for complex workflows")
    print("   4. Test error handling and exception paths")

def main():
    """Main function to orchestrate coverage report generation."""
    print("🚀 CUDA Module Code Coverage Report Generator")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("logic/cuda").exists():
        print("❌ Error: This script should be run from the project root directory")
        print("   Expected to find 'logic/cuda' directory")
        sys.exit(1)
    
    # Install coverage tools
    install_coverage_tools()
    
    # Run coverage tests
    success = run_coverage_tests()
    
    # Generate detailed reports
    generate_detailed_coverage_report()
    
    # Analyze results
    analyze_coverage_results()
    
    if success:
        print("\n✅ Coverage report generation completed successfully!")
        print("📊 Check the generated reports for detailed coverage analysis")
    else:
        print("\n⚠️ Some tests failed or coverage is below threshold")
        print("📊 Review the coverage report to identify areas for improvement")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 