# Harbour

Harbour is the order-intake service. It accepts orders from the storefront,
validates them against inventory, and publishes them to the fulfilment queue.

## Build

Build the service with:

    make build

Run the test suite with `make test`. Both targets expect Python 3.11 or newer.

## Layout

- `src/` — the service itself
- `data/` — sample fixtures used by the tests
- `docs/` — the runbook and the on-call rota
- `config/` — environment settings

## Owners

Harbour is maintained by the Platform team. Escalation path is in
`docs/runbook.md`.
