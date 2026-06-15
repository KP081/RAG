"""
Security Layer
Input senitization, PII detection/masking, output validation.
"""

import re
from typing import Optional
from langsmith import traceable


# Input Sanitization
class Inputsanitizer:
    SYSTEM_PROMPT_PATTERNS = [
        r"ignore\s+previous\s+instructions?",
        r"forget\s+all\s+instructions?",
        r"reveal\s+your\s+system\s+prompt",
        r"show\s+me\s+your\s+system\s+prompt",
        r"print\s+your\s+instructions?",
        r"display\s+hidden\s+prompt",
    ]

    OVERRIDE_PATTERNS = [
        r"ignore\s+everything\s+above",
        r"new\s+instructions?",
        r"from\s+now\s+on",
        r"override\s+previous\s+instructions?",
        r"replace\s+your\s+instructions?",
    ]

    ROLE_PATTERNS = [
        r"act\s+as",
        r"pretend\s+you\s+are",
        r"roleplay\s+as",
        r"simulate\s+a",
        r"you\s+are\s+now",
    ]

    DATA_EXFIL_PATTERNS = [
        r"dump\s+all\s+documents?",
        r"show\s+complete\s+context",
        r"print\s+retrieved\s+chunks?",
        r"return\s+raw\s+database",
        r"show\s+source\s+documents?",
    ]

    LEAKAGE_PATTERNS = [
        r"show\s+hidden\s+context",
        r"display\s+context",
        r"what\s+information\s+was\s+retrieved",
        r"print\s+retrieved\s+text",
    ]

    JAILBREAK_PATTERNS = [
        r"\bdan\b",
        r"do\s+anything\s+now",
        r"developer\s+mode",
        r"god\s+mode",
        r"unrestricted\s+mode",
        r"jailbreak",
    ]

    TOOL_PATTERNS = [
        r"call\s+tool",
        r"execute\s+tool",
        r"invoke\s+tool",
        r"run\s+command",
        r"execute\s+bash",
    ]

    ENCODING_PATTERNS = [
        r"base64",
        r"hex",
        r"rot13",
        r"unicode\s+escape",
        r"encoded\s+prompt",
    ]

    SQLI_PATTERNS = [
        r"union\s+select",
        r"drop\s+table",
        r"delete\s+from",
        r"insert\s+into",
        r"or\s+1\s*=\s*1",
    ]

    INJECTION_PATTERNS = [
        *SYSTEM_PROMPT_PATTERNS,
        *OVERRIDE_PATTERNS,
        *ROLE_PATTERNS,
        *DATA_EXFIL_PATTERNS,
        *LEAKAGE_PATTERNS,
        *JAILBREAK_PATTERNS,
        *TOOL_PATTERNS,
        *ENCODING_PATTERNS,
        *SQLI_PATTERNS,
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """
        check if input is safe.
        returns: (is_safe, rejection_reason)
        """
        for pattern in self.patterns:
            if pattern.search(text):
                return False, "Blocked: potential prompt injection detected!"
        return True, None

    def clean(self, text: str) -> str:
        """Remove potentially dangerous delimiters from input."""
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        text = text.replace("{{", "{ {").replace("}}", "} }")
        return text.strip()


class PIIDetector:
    """
    Detect and mask Personally Identifiable Information (PII).

    Works on:
    - User Input -> Before sending to LLM
    - LLM Output -> Before returning to Client
    """

    PATTERNS = {
        # Email
        "EMAIL": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        # Indian Phone Number
        "PHONE": re.compile(r"(?<!\w)(?:\+91[\s\-]?)?[6-9]\d{9}(?!\w)"),
        # Aadhaar
        "AADHAAR": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
        # PAN
        "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        # Indian Passport
        "PASSPORT": re.compile(r"\b[A-Z][0-9]{7}\b"),
        # Driving License
        "DRIVING_LICENSE": re.compile(
            r"\b[A-Z]{2}[0-9]{2}\s?[0-9]{11}\b",
            re.IGNORECASE,
        ),
        # Credit Card
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
        # IFSC
        "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        # UPI
        "UPI": re.compile(
            r"\b[a-zA-Z0-9._-]{2,}@(?:upi|ybl|ibl|axl|oksbi|okhdfcbank|paytm|apl)\b",
            re.IGNORECASE,
        ),
        # IPv4
        "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        # IPv6
        "IPV6": re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
        # US SSN
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        # DOB
        "DOB": re.compile(
            r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b"
        ),
        # URL
        "URL": re.compile(
            r"https?://[^\s]+",
            re.IGNORECASE,
        ),
        # JWT
        "JWT": re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        # AWS Access Key
        "AWS_ACCESS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        # GitHub Token
        "GITHUB_TOKEN": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        # OpenAI Key
        "OPENAI_API_KEY": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        # Generic API Key
        "API_KEY": re.compile(
            r'(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{16,}["\']?'
        ),
    }

    MASK_MAP = {
        "EMAIL": "[EMAIL_REDACTED]",
        "PHONE": "[PHONE_REDACTED]",
        "AADHAAR": "[AADHAAR_REDACTED]",
        "PAN": "[PAN_REDACTED]",
        "PASSPORT": "[PASSPORT_REDACTED]",
        "DRIVING_LICENSE": "[DRIVING_LICENSE_REDACTED]",
        "CREDIT_CARD": "[CREDIT_CARD_REDACTED]",
        "IFSC": "[IFSC_REDACTED]",
        "UPI": "[UPI_REDACTED]",
        "IPV4": "[IPV4_REDACTED]",
        "IPV6": "[IPV6_REDACTED]",
        "SSN": "[SSN_REDACTED]",
        "DOB": "[DOB_REDACTED]",
        "URL": "[URL_REDACTED]",
        "JWT": "[JWT_REDACTED]",
        "AWS_ACCESS_KEY": "[AWS_ACCESS_KEY_REDACTED]",
        "GITHUB_TOKEN": "[GITHUB_TOKEN_REDACTED]",
        "OPENAI_API_KEY": "[OPENAI_API_KEY_REDACTED]",
        "API_KEY": "[API_KEY_REDACTED]",
    }

    # Specific patterns first, generic patterns last
    MASK_ORDER = [
        "EMAIL",
        "AADHAAR",
        "PAN",
        "PASSPORT",
        "DRIVING_LICENSE",
        "CREDIT_CARD",
        "IFSC",
        "UPI",
        "IPV4",
        "IPV6",
        "SSN",
        "DOB",
        "URL",
        "JWT",
        "AWS_ACCESS_KEY",
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "API_KEY",
        "PHONE",
    ]

    def detect(self, text: str) -> dict[str, list[str]]:
        """
        Detect all PII in text.
        """
        findings = {}

        for pii_type, pattern in self.PATTERNS.items():
            matches = [m.group(0) for m in pattern.finditer(text)]

            if matches:
                findings[pii_type] = matches

        return findings

    def mask(self, text: str) -> str:
        """
        Replace detected PII with redaction markers.
        """
        masked = text

        for pii_type in self.MASK_ORDER:
            pattern = self.PATTERNS[pii_type]
            replacement = self.MASK_MAP[pii_type]

            masked = pattern.sub(replacement, masked)

        return masked


# Output validation
class OutputValidator:
    """
    Validate LLM Output before returning to the client.
    Caches PII leakage and harmful content in responses.
    """
    
    HARMFUL_PATTERNS = [
        re. compile(r"here('s| is) (how| the way) to (hack|steal|attack)", re.I),
        re. compile(r"password\s+is\s+", re.I),
        re.compile(r"api[_\s-]?key\s*[:=]", re.I),
    ]
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        
    def validate(self, output: str) -> tuple[str, list[str]]:
        """
        validate and clean output.
        Returns: (cleaned_output, list_of_warnings)
        """
        warnings = []
        
        # check PII leakage in output
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")
            
        # check for harmful content
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output = "[Response Blocked: potentially harmful content]"
                warnings.append("Harmful Content Blocked")
                break
            
        return output, warnings
    
    
class SecurityPipeline:
    """
    Full security pipeline that processes input and output.
    this is a single class wire into in API.
    """
    
    def __init__(self):
        self.sanitizer = Inputsanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()
        
    @traceable(name="security_check_point")
    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        """
        Process input through security checks.
        Returns: (is_allowd, cleaned_text, security_notes)
        """
        notes = []
        
        # step 1: check for injection
        is_safe, reason = self.sanitizer.check(text)
        if not is_safe:
            return False, "", [reason]
        
        # step 2: clean input
        cleaned = self.sanitizer.clean(text)
        
        # step 3: mask PII before it reaches the LLM
        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f"input PII masked: {list(pii_found.keys())}")
            
        return True, cleaned, notes
    
    @traceable(name="security_check_output")
    def check_output(self, text: str) -> tuple[str, list[str]]:
        """
        validate output before returning to client.
        returns: (cleand output, warnings)
        """
        return self.output_validator.validate(text)