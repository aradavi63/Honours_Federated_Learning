"""Validate and copy server-visible models for attack evaluation."""

import torch

from schemes import ENCRYPTED_SCHEMES
from training.setup import _clone_state


def capture_release(
    scheme, round_number, start, individual, aggregate, client_ids, routes=None
):
    expected = set(client_ids)
    if scheme in ENCRYPTED_SCHEMES:
        if individual:
            raise ValueError(
                "Encrypted server evidence must not contain individual models"
            )
        aggregate_boundary = "ServerNodeClass.decryptWeights:after_mask_removal"
    else:
        if set(individual) != expected:
            raise ValueError("Missing or unexpected server-received client releases")
        aggregate_boundary = "ServerNodeClass.trainingProcess:after_FedAvg_model_load"
    for state in [aggregate, *individual.values()]:
        if set(state) != set(start) or any(
            state[n].shape != start[n].shape or not torch.isfinite(state[n]).all()
            for n in start
        ):
            raise ValueError("Invalid released model state")
    metadata = dict(
        schema_version=1,
        scheme=scheme,
        round=round_number,
        source="actual_server_observation",
        attacker="passive_non_colluding_server",
        client_ids=sorted(expected),
        individual_client_ids=sorted(individual),
        round_start_boundary="ServerNodeClass.updateOverheadDict:before_client_training",
        individual_boundary="ServerNodeClass.receiveMessage:RESULTS[0]"
        if individual
        else None,
        aggregate_boundary=aggregate_boundary,
        aggregate_scope="all_participating_clients",
        individual_scope="single_client" if individual else "unavailable",
        routes=routes or [],
        coverage="Model-release view; not a complete transcript of route ciphertexts, scalar losses, noise messages or key material",
        update_convention="released_model_minus_round_start_model",
    )
    payload = dict(
        metadata=metadata,
        round_start=_clone_state(start),
        individual={c: _clone_state(v) for c, v in individual.items()},
        aggregate=_clone_state(aggregate),
    )
    return payload
