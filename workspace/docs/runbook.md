# Harbour runbook

## On-call

Primary on-call sits with the Platform team in the Helsinki office. Secondary
cover is provided from Toronto during North American hours.

## Common alerts

### QueueDepthHigh

The fulfilment queue is backing up. Check the publisher logs first; the usual
cause is a downstream timeout, not Harbour itself.

### DuplicateOrderRate

Fires when more than 2% of orders in a five-minute window are rejected as
duplicates. This almost always means the storefront is retrying too
aggressively.

## Restarting

Harbour is stateless. Restart with `make restart`; there is no drain step.
