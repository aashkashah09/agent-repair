# Interface defect catalogue

Eight recurring defect patterns found by auditing published MCP servers, and the
24 graduated instances seeded into the commerce-and-support server used for
evaluation. The machine-readable form is `data/defects/seeded_defects.json`;
`toolsmith seed` applies it to `data/schemas/clean` to produce
`data/schemas/seeded`.

A defect here is never a broken tool. Every implementation in
`src/toolsmith/server/tools.py` works, and none of them change when a defect is
seeded or repaired. What changes is what the published schema says about the
tool: its description, its parameter descriptions, its declared types and
value sets, and the error returns it documents. Two patterns cannot be
expressed that way — a tool that swallows an error condition and one that
truncates a result have to actually behave that way — so those carry a runtime
wrapper alongside their schema mutations.

## Severity

Each pattern is seeded three times, once at each severity. Severity is how much
of the constraint survives in the published schema, not how much damage the
defect does:

| Severity | The schema |
|---|---|
| **mild** | states the constraint but is stale, incomplete, or imprecise about it |
| **moderate** | gestures at the constraint without giving a caller enough to act on |
| **severe** | says nothing about it; the constraint is discoverable only by failing |

Grading severity separately from impact is what makes the repair results
readable. A defect the task suite rarely reaches produces few failures whatever
its severity, and the loop has correspondingly little to work with.

---

## P1 · undocumented-enum

A parameter accepts a closed set of values, and the schema publishes it as an
open one — or publishes a set that no longer matches what the server takes. The
agent invents a value that reads naturally, the call is rejected, and the
rejection names no alternatives, so the next attempt is another guess.

| | Instance | Where |
|---|---|---|
| severe | `D01` | `cancel_order.reason` — four accepted literals, declared as an unconstrained string with a description that invites prose |
| moderate | `D02` | `create_support_ticket.category` — the enum still advertises a retired queue and omits `return_request`, which the server accepts |
| mild | `D03` | `update_support_ticket.priority` — the enum omits `urgent`, which the server accepts and which agents reach for on escalations |

## P2 · silent-empty-return

An error condition returns a well-formed empty result instead of an error. The
agent cannot distinguish "nothing matched" from "your call was wrong", and the
empty result is the more natural reading, so it reports absence to the customer.

| | Instance | Where |
|---|---|---|
| severe | `D04` | `search_products` — an unrecognised `category` returns `{"results": [], "total_matches": 0}` |
| moderate | `D05` | `track_shipment` — an order that has not shipped returns a record with a null status and no events |
| mild | `D06` | `list_orders` — an unknown `customer_id` returns an empty order list |

## P3 · ambiguous-datetime

A time value crosses the interface with no format attached. On input the server
rejects everything but one form; on output the agent compares values that are
not in the same form and draws a conclusion that is wrong by an unstated
amount.

| | Instance | Where |
|---|---|---|
| severe | `D07` | `list_orders.since` — documented as a date, requires an ISO-8601 instant with an offset, and the rejection is not among the documented error returns |
| moderate | `D08` | `check_inventory.restock_eta` — returned as a bare calendar date where every other timestamp in the domain is an instant |
| mild | `D09` | `track_shipment.estimated_delivery` — described as a timestamp, returned as a bare date |

## P4 · overloaded-tool-mode

One tool covers several distinct operations behind a flag, and each operation
reads a different set of fields from the same argument. The schema documents the
union, so a call that is well-formed for one mode is missing a field in
another.

| | Instance | Where |
|---|---|---|
| severe | `D10` | `modify_order_items.mode` — add, remove and replace each need different fields on `items`, and the schema distinguishes none of them |
| moderate | `D11` | `update_support_ticket` — status change, priority change and note append are three operations behind one call, with no statement that at least one is required |
| mild | `D12` | `process_refund.amount` — omitting it switches a partial refund to a full one; the schema presents it as an ordinary optional field |

## P5 · unstated-precondition

The call requires state that an earlier call has to establish. Nothing in the
schema says so, and the rejection does not name what is missing, so the agent
retries the same call rather than doing the thing that would make it valid.

| | Instance | Where |
|---|---|---|
| severe | `D13` | `process_refund` — the order must already be cancelled or returned; neither the description nor the error returns mention it |
| moderate | `D14` | `modify_order_items` — only pending orders can be modified; a state error exists but does not say which states are permitted |
| mild | `D15` | `create_order.address_id` — optional only on an account with exactly one saved address |

## P6 · near-duplicate-tool-naming

Two tools describe themselves in terms that do not separate them. Each
description is accurate on its own; read together they give no basis for
choosing, so selection becomes a coin flip and the wrong result is plausible
enough to act on.

This is the one pattern that is not a property of a single interface, which is
why a revision to either tool alone can improve it without resolving it.

| | Instance | Where |
|---|---|---|
| severe | `D16` | `get_product_detail` and `check_inventory` — both describe themselves as returning product information including availability |
| moderate | `D17` | `search_products` and `get_product_detail` — search advertises product details, so the follow-up that resolves SKUs looks redundant |
| mild | `D18` | `create_support_ticket` and `update_support_ticket` — create describes itself as creating *or updating* a ticket |

## P7 · untyped-passthrough

A parameter is declared as a string or a bare object where the server requires
structure. The schema is satisfied by a value the implementation cannot use, so
validation passes and the call fails — or, worse, succeeds and stores something
unusable.

| | Instance | Where |
|---|---|---|
| severe | `D19` | `update_customer_profile.address` — typed as a string; the server requires an object with `line1`, `city`, `state` and `postal_code` |
| moderate | `D20` | `create_order.items` — an array whose element type is `object` with no properties, leaving `sku` and `quantity` undiscoverable |
| mild | `D21` | `search_products.max_price` — typed as a string although it is compared numerically against catalogue prices |

## P8 · silent-pagination-truncation

A result is capped and the response carries nothing that reveals the cap. The
agent reasons over a partial set as though it were complete, and concludes that
a record does not exist because it was not on the page it read.

| | Instance | Where |
|---|---|---|
| severe | `D22` | `list_orders` — one page, no cursor, no match count; an order past the twentieth is indistinguishable from one that does not exist |
| moderate | `D23` | `search_products` — results truncated to ten regardless of `limit`, with `total_matches` withheld |
| mild | `D24` | `track_shipment.events` — only the two most recent scans are returned |

---

## Audit sources

The patterns were collected by reading the published tool schemas of open MCP
servers across file, database, issue-tracker, calendar, payments, and
cloud-resource domains, and comparing each schema against the behaviour of the
implementation behind it. Every pattern here appeared in more than one server,
in more than one domain; single-server oddities were left out. The seeded
instances are written against this server's own tools rather than lifted
verbatim, so the catalogue transfers and the instances do not.

Two things the audit turned up that are not represented here. Cross-tool
ordering constraints — where the second of two calls is only valid after the
first, and neither schema mentions the other — appear frequently but resist
seeding as an instance of a single tool. Interfaces whose behaviour depends on
server-side configuration invisible to the caller appear in the cloud-resource
servers and cannot be reproduced in a self-contained evaluation server at all.
