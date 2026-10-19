"""Lossless JSON representation of special floating-point input parameters."""
import math


def encode_special_floats(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'__float__': str(value)}
    if isinstance(value, dict):
        return {key: encode_special_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_special_floats(item) for item in value]
    return value


def decode_special_floats(value):
    if isinstance(value, dict):
        if set(value) == {'__float__'} and value['__float__'] in ('inf', '-inf', 'nan'):
            return float(value['__float__'])
        return {key: decode_special_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_special_floats(item) for item in value]
    return value
