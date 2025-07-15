"""
Steganan

Steganography in floating point data with NaN payloads
"""

__author__ = "Kelsey Jordahl"
__version__ = "0.2.0"
__all__ = [
    "decode_array",
    "encode_array",
    "is_payload_nan",
    "retrieve_string_from_payloads",
    "str_to_bytes_array",
    "str_to_uint32_array",
    "uint32_array_to_str",
    "write_str_to_nans",
]

from .steganan import (
    decode_array,
    encode_array,
    is_payload_nan,
    retrieve_string_from_payloads,
    str_to_bytes_array,
    str_to_uint32_array,
    uint32_array_to_str,
    write_str_to_nans,
)
