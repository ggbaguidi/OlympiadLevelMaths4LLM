# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,invalid-name
"""Tool output verification and validation.

This module provides verification capabilities:
- Numerical result validation (NaN, Infinity, bounds checking)
- Error pattern detection and classification
- Output format validation
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

@dataclass
class VerificationResult:
    """Result of verifying a tool output."""

    is_valid: bool
    confidence: float  # 0.0 to 1.0
    error_type: str | None = None
    error_details: str | None = None
    numerical_value: float | int | None = None
    warnings: list[str] = field(default_factory=list)


class ToolOutputVerifier:
    """Verifies tool outputs for correctness and validity."""

    # Patterns for error detection
    ERROR_PATTERNS = {
        "syntax_error": re.compile(r"SyntaxError|syntax error", re.IGNORECASE),
        "name_error": re.compile(r"NameError|name.*not defined", re.IGNORECASE),
        "type_error": re.compile(r"TypeError", re.IGNORECASE),
        "value_error": re.compile(r"ValueError", re.IGNORECASE),
        "zero_division": re.compile(
            r"ZeroDivisionError|division by zero", re.IGNORECASE
        ),
        "index_error": re.compile(r"IndexError|index out of range", re.IGNORECASE),
        "key_error": re.compile(r"KeyError", re.IGNORECASE),
        "attribute_error": re.compile(r"AttributeError", re.IGNORECASE),
        "import_error": re.compile(r"ImportError|ModuleNotFoundError", re.IGNORECASE),
        "timeout": re.compile(r"timed out|timeout", re.IGNORECASE),
        "memory_error": re.compile(r"MemoryError|out of memory", re.IGNORECASE),
        "recursion_error": re.compile(
            r"RecursionError|maximum recursion depth", re.IGNORECASE
        ),
    }

    # Pattern to extract numerical results
    NUMBER_PATTERN = re.compile(
        r"(?:result|answer|value|output)\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|"
        r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:result|answer|value|output)|"
        r"\\boxed\{(-?\d+)\}",
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_value: float = -1e15,
        max_value: float = 1e15,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf

    def verify_output(
        self, output: str, expected_answer: int | None = None
    ) -> VerificationResult:
        """Verify tool output for correctness.

        Args:
            output: The tool output string
            expected_answer: Optional expected answer to compare against

        Returns:
            VerificationResult with validation details
        """
        warnings = []

        # Check for errors
        error_type, error_details = self._detect_errors(output)
        if error_type:
            return VerificationResult(
                is_valid=False,
                confidence=0.0,
                error_type=error_type,
                error_details=error_details,
                warnings=warnings,
            )

        # Extract numerical value
        numerical_value = self._extract_numerical_value(output)

        # Validate numerical value
        if numerical_value is not None:
            is_valid, validation_warnings = self._validate_numerical_value(
                numerical_value
            )
            warnings.extend(validation_warnings)

            if not is_valid:
                return VerificationResult(
                    is_valid=False,
                    confidence=0.0,
                    error_type="invalid_numerical_value",
                    error_details=f"Numerical value {numerical_value} failed validation",
                    numerical_value=numerical_value,
                    warnings=warnings,
                )

            # Check against expected answer if provided
            if expected_answer is not None:
                if abs(numerical_value - expected_answer) < 1e-9:
                    return VerificationResult(
                        is_valid=True,
                        confidence=1.0,
                        numerical_value=numerical_value,
                        warnings=warnings,
                    )
                else:
                    warnings.append(
                        f"Numerical value {numerical_value} doesn't match expected {expected_answer}"
                    )

        # Check for verification markers
        confidence = self._calculate_confidence(output, numerical_value is not None)

        return VerificationResult(
            is_valid=confidence > 0.5,
            confidence=confidence,
            numerical_value=numerical_value,
            warnings=warnings,
        )

    def _detect_errors(self, output: str) -> tuple[str | None, str | None]:
        """Detect and classify errors in output.

        Returns:
            Tuple of (error_type, error_details) or (None, None) if no error
        """
        # Check for traceback
        if "Traceback (most recent call last):" in output:
            # Extract the error line
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "Traceback" in line:
                    # Look for the actual error in the next few lines
                    for j in range(i + 1, min(i + 10, len(lines))):
                        for error_name, pattern in self.ERROR_PATTERNS.items():
                            if pattern.search(lines[j]):
                                # Get the error message
                                error_msg = lines[j].strip()
                                if j + 1 < len(lines):
                                    error_msg += " " + lines[j + 1].strip()
                                return error_name, error_msg[:200]
            return "runtime_error", "Python runtime error (traceback detected)"

        # Check for [ERROR] markers
        if "[ERROR]" in output:
            for error_name, pattern in self.ERROR_PATTERNS.items():
                if pattern.search(output):
                    return error_name, output[:200]
            return "unknown_error", "Error marker found in output"

        # Check for ERROR patterns without traceback
        for error_name, pattern in self.ERROR_PATTERNS.items():
            if pattern.search(output):
                return error_name, output[:200]

        return None, None

    def _extract_numerical_value(self, output: str) -> float | int | None:
        """Extract numerical result from output."""
        # Try to find boxed answer first
        boxed_match = re.search(r"\\boxed\{(-?\d+)\}", output)
        if boxed_match:
            try:
                return int(boxed_match.group(1))
            except ValueError:
                pass

        # Look for explicit result statements
        matches = self.NUMBER_PATTERN.findall(output)
        for match in matches:
            for group in match:
                if group:
                    try:
                        val = float(group)
                        # Return as int if it's a whole number
                        if val == int(val):
                            return int(val)
                        return val
                    except ValueError:
                        continue

        # Fallback: find any number that looks like a result
        # Look for lines with numbers and result-like keywords
        for line in output.split("\n"):
            if any(
                keyword in line.lower()
                for keyword in ["result", "answer", "value", "output"]
            ):
                numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                if numbers:
                    try:
                        val = float(numbers[-1])  # Take the last number on the line
                        if val == int(val):
                            return int(val)
                        return val
                    except ValueError:
                        pass

        return None

    def _validate_numerical_value(self, value: float | int) -> tuple[bool, list[str]]:
        """Validate a numerical value.

        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []

        # Check for NaN
        if isinstance(value, float) and math.isnan(value):
            if not self.allow_nan:
                return False, ["Value is NaN (Not a Number)"]
            warnings.append("Value is NaN")

        # Check for Infinity
        if isinstance(value, float) and math.isinf(value):
            if not self.allow_inf:
                return False, ["Value is Infinity"]
            warnings.append("Value is Infinity")

        # Check bounds
        if value < self.min_value:
            return False, [f"Value {value} is below minimum {self.min_value}"]
        if value > self.max_value:
            return False, [f"Value {value} is above maximum {self.max_value}"]

        # Check for very small or very large numbers (suspicious)
        abs_val = abs(value)
        if abs_val > 0 and (abs_val < 1e-10 or abs_val > 1e10):
            warnings.append(f"Value {value} has extreme magnitude")

        return True, warnings

    def _calculate_confidence(self, output: str, has_numerical: bool) -> float:
        """Calculate confidence score for the output."""
        confidence = 0.5  # Base confidence

        # Boost confidence if we have numerical result
        if has_numerical:
            confidence += 0.2

        # Boost for verification markers
        if "VERIFY_OK" in output:
            confidence += 0.3

        # Boost for explicit success indicators
        success_indicators = [
            "success",
            "completed",
            "finished",
            "done",
            "result:",
            "answer:",
        ]
        for indicator in success_indicators:
            if indicator in output.lower():
                confidence += 0.1
                break

        # Penalize for warning indicators
        warning_indicators = [
            "warning",
            "caution",
            "attention",
            "note:",
        ]
        for indicator in warning_indicators:
            if indicator in output.lower():
                confidence -= 0.1
                break

        # Check for incomplete execution indicators
        incomplete_indicators = [
            "...",
            "continued",
            "truncated",
            "[output truncated]",
        ]
        for indicator in incomplete_indicators:
            if indicator in output.lower():
                confidence -= 0.2
                break

        return max(0.0, min(1.0, confidence))
