"""Scheme identities shared by training, attacks and reporting."""

SCHEMES = ("fedavg", "sihefl", "dp_fedavg", "dp_sihefl")
ENCRYPTED_SCHEMES = frozenset(("sihefl", "dp_sihefl"))
PRIVATE_SCHEMES = frozenset(("dp_fedavg", "dp_sihefl"))
