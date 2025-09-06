"""
Test configuration and utilities for the trading strategy ML system.
"""

import os
import sys
import unittest
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestConfig:
    """Configuration for test execution."""
    
    # Test categories
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    PERFORMANCE_TESTS = "performance_tests"
    
    # Test directories
    TEST_DIRS = {
        UNIT_TESTS: "tests/unit_tests",
        INTEGRATION_TESTS: "tests/integration_tests",
        PERFORMANCE_TESTS: "tests/performance_tests"
    }
    
    # Test files
    TEST_FILES = {
        UNIT_TESTS: ["test_components.py"],
        INTEGRATION_TESTS: ["test_integration.py"],
        PERFORMANCE_TESTS: ["test_performance.py"]
    }
    
    # Performance thresholds (in seconds)
    PERFORMANCE_THRESHOLDS = {
        "indicator_calculation": 10.0,
        "feature_generation": 5.0,
        "ml_training": 30.0,
        "backtest_execution": 30.0,
        "performance_analysis": 15.0
    }
    
    # Memory thresholds (in MB)
    MEMORY_THRESHOLDS = {
        "indicator_calculation": 500,
        "feature_generation": 200,
        "ml_model_creation": 300,
        "backtest_execution": 1000
    }


class TestRunner:
    """Test runner for executing different types of tests."""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results = {}
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        logger.info("Running unit tests...")
        
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Load unit tests
        unit_test_dir = self.config.TEST_DIRS[self.config.UNIT_TESTS]
        for test_file in self.config.TEST_FILES[self.config.UNIT_TESTS]:
            test_path = os.path.join(unit_test_dir, test_file)
            if os.path.exists(test_path):
                # Import test module
                module_name = test_file.replace('.py', '')
                spec = unittest.util.spec_from_file_location(module_name, test_path)
                test_module = unittest.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                
                # Add tests to suite
                loader = unittest.TestLoader()
                tests = loader.loadTestsFromModule(test_module)
                test_suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Store results
        self.results[self.config.UNIT_TESTS] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful(),
            'failures_list': result.failures,
            'errors_list': result.errors
        }
        
        logger.info(f"Unit tests completed: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        return self.results[self.config.UNIT_TESTS]
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Load integration tests
        integration_test_dir = self.config.TEST_DIRS[self.config.INTEGRATION_TESTS]
        for test_file in self.config.TEST_FILES[self.config.INTEGRATION_TESTS]:
            test_path = os.path.join(integration_test_dir, test_file)
            if os.path.exists(test_path):
                # Import test module
                module_name = test_file.replace('.py', '')
                spec = unittest.util.spec_from_file_location(module_name, test_path)
                test_module = unittest.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                
                # Add tests to suite
                loader = unittest.TestLoader()
                tests = loader.loadTestsFromModule(test_module)
                test_suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Store results
        self.results[self.config.INTEGRATION_TESTS] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful(),
            'failures_list': result.failures,
            'errors_list': result.errors
        }
        
        logger.info(f"Integration tests completed: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        return self.results[self.config.INTEGRATION_TESTS]
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        logger.info("Running performance tests...")
        
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Load performance tests
        performance_test_dir = self.config.TEST_DIRS[self.config.PERFORMANCE_TESTS]
        for test_file in self.config.TEST_FILES[self.config.PERFORMANCE_TESTS]:
            test_path = os.path.join(performance_test_dir, test_file)
            if os.path.exists(test_path):
                # Import test module
                module_name = test_file.replace('.py', '')
                spec = unittest.util.spec_from_file_location(module_name, test_path)
                test_module = unittest.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                
                # Add tests to suite
                loader = unittest.TestLoader()
                tests = loader.loadTestsFromModule(test_module)
                test_suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Store results
        self.results[self.config.PERFORMANCE_TESTS] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful(),
            'failures_list': result.failures,
            'errors_list': result.errors
        }
        
        logger.info(f"Performance tests completed: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        return self.results[self.config.PERFORMANCE_TESTS]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        logger.info("Running all tests...")
        
        # Run all test categories
        unit_results = self.run_unit_tests()
        integration_results = self.run_integration_tests()
        performance_results = self.run_performance_tests()
        
        # Calculate overall results
        total_tests = unit_results['tests_run'] + integration_results['tests_run'] + performance_results['tests_run']
        total_failures = unit_results['failures'] + integration_results['failures'] + performance_results['failures']
        total_errors = unit_results['errors'] + integration_results['errors'] + performance_results['errors']
        
        self.results['overall'] = {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'success_rate': (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"All tests completed: {total_tests} tests, {total_failures} failures, {total_errors} errors")
        return self.results
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        report = f"""
# Test Execution Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

"""
        
        if 'overall' in self.results:
            overall = self.results['overall']
            report += f"""
- **Total Tests**: {overall['total_tests']}
- **Failures**: {overall['total_failures']}
- **Errors**: {overall['total_errors']}
- **Success Rate**: {overall['success_rate']:.2%}

"""
        
        # Detailed results for each category
        for category in [self.config.UNIT_TESTS, self.config.INTEGRATION_TESTS, self.config.PERFORMANCE_TESTS]:
            if category in self.results:
                results = self.results[category]
                report += f"""
## {category.replace('_', ' ').title()}

- **Tests Run**: {results['tests_run']}
- **Failures**: {results['failures']}
- **Errors**: {results['errors']}
- **Success**: {'Yes' if results['success'] else 'No'}

"""
                
                # List failures
                if results['failures'] > 0:
                    report += "### Failures\n"
                    for test, traceback in results['failures_list']:
                        report += f"- **{test}**: {traceback}\n"
                
                # List errors
                if results['errors'] > 0:
                    report += "### Errors\n"
                    for test, traceback in results['errors_list']:
                        report += f"- **{test}**: {traceback}\n"
        
        return report
    
    def save_results(self, filename: str = None):
        """Save test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_results_{timestamp}.json"
        
        # Create results directory
        os.makedirs("test_results", exist_ok=True)
        filepath = os.path.join("test_results", filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filepath}")
        
        # Save report
        report = self.generate_test_report()
        report_filename = filename.replace('.json', '.md')
        report_filepath = os.path.join("test_results", report_filename)
        
        with open(report_filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Test report saved to {report_filepath}")


def run_tests(test_category: str = "all") -> Dict[str, Any]:
    """Quick function to run tests."""
    runner = TestRunner()
    
    if test_category == "unit":
        return runner.run_unit_tests()
    elif test_category == "integration":
        return runner.run_integration_tests()
    elif test_category == "performance":
        return runner.run_performance_tests()
    elif test_category == "all":
        return runner.run_all_tests()
    else:
        raise ValueError(f"Unknown test category: {test_category}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for the trading strategy ML system")
    parser.add_argument("--category", choices=["unit", "integration", "performance", "all"], 
                       default="all", help="Test category to run")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Run tests
    runner = TestRunner()
    
    if args.category == "unit":
        results = runner.run_unit_tests()
    elif args.category == "integration":
        results = runner.run_integration_tests()
    elif args.category == "performance":
        results = runner.run_performance_tests()
    else:
        results = runner.run_all_tests()
    
    # Save results if requested
    if args.save:
        runner.save_results()
    
    # Print summary
    print(f"\nTest execution completed!")
    print(f"Category: {args.category}")
    
    if 'overall' in results:
        overall = results['overall']
        print(f"Total tests: {overall['total_tests']}")
        print(f"Failures: {overall['total_failures']}")
        print(f"Errors: {overall['total_errors']}")
        print(f"Success rate: {overall['success_rate']:.2%}")
    else:
        print(f"Tests run: {results['tests_run']}")
        print(f"Failures: {results['failures']}")
        print(f"Errors: {results['errors']}")
        print(f"Success: {'Yes' if results['success'] else 'No'}")
