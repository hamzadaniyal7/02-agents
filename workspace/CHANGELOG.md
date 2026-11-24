# Changelog

## 2.4.0 — 2026-09-02
- Orders are now deduplicated by idempotency key before validation.
- The fulfilment publisher retries three times with exponential backoff.

## 2.3.1 — 2026-07-18
- Fixed a crash when an order arrived with no line items.

## 2.3.0 — 2026-06-30
- Added the `refunded` order status.
- Inventory checks moved off the request path.
