"""
backend/rag/sample_contracts/contract_text.py
──────────────────────────────────────────────────────────────────────────────
Five realistic procurement contract texts used to seed the FAISS index during
development/testing.  Each contract intentionally contains specific clauses
that trigger different fraud patterns when cross-referenced by the LLM layer.

Contracts
---------
1. Apex IT Solutions         – IT Hardware & Support Services
2. BuildRight Construction   – Facility Renovation Services
3. MedSupply Corp            – Medical Equipment & Supplies
4. LogiTrans Freight         – Logistics & Transportation
5. CloudScale Consulting     – Professional Consulting Services
"""

from __future__ import annotations
from typing import Dict

SAMPLE_CONTRACTS: Dict[str, Dict[str, str]] = {
    "apex_it": {
        "title": "IT Hardware and Support Services Agreement – Apex IT Solutions",
        "vendor_name": "Apex IT Solutions LLC",
        "text": """
IT HARDWARE AND SUPPORT SERVICES AGREEMENT

CONTRACT NUMBER: IT-2024-0081
EFFECTIVE DATE: January 1, 2024
EXPIRATION DATE: December 31, 2025
TOTAL CONTRACT VALUE: $2,400,000.00

PARTIES
This Agreement is entered into between the Federal Procurement Agency ("Agency")
and Apex IT Solutions LLC ("Vendor"), a corporation registered in Delaware.

1. SCOPE OF SERVICES

1.1 Hardware Procurement
Vendor shall supply server hardware, networking equipment, and end-user devices
as specified in Exhibit A. All hardware must be new, unused, and carry manufacturer
warranty for a minimum of 36 months from the date of delivery.

1.2 Support Services
Vendor shall provide Tier 1 and Tier 2 IT support services on-site during
business hours (Monday–Friday, 08:00–18:00 Eastern Time). Emergency on-call
support is available 24/7 for Priority-1 incidents.

1.3 Cloud Migration Services
Vendor shall provide up to 500 person-hours of cloud migration consulting per
contract year. Additional hours may be purchased at the rate of $185 per hour,
subject to a written Task Order approved by the Contracting Officer.

2. PRICING TERMS

2.1 Fixed Unit Prices
All hardware items are sold at the unit prices listed in Schedule B. Prices are
firm-fixed and shall not increase during the contract period without a formal
contract modification signed by both parties.

2.2 Invoice Submission and Payment
Vendor shall submit invoices no more than once per calendar month per service
category. Invoices submitted more frequently than monthly shall be consolidated
and returned unpaid. The Agency shall pay undisputed invoices within 30 days of
receipt. Late payments accrue interest at 1.5% per month.

2.3 Split Invoice Prohibition
Vendor is expressly prohibited from splitting a single deliverable into multiple
invoices to circumvent the Simplified Acquisition Threshold of $10,000. Any
invoice that appears to split a single procurement shall be reported to the
Inspector General and may result in contract termination.

2.4 Approval Thresholds
Individual purchase orders exceeding $25,000 require prior written approval from
the Deputy Contracting Officer. Orders exceeding $100,000 require approval from
the Chief Procurement Officer. Orders exceeding $250,000 require Agency Executive
approval and must be published in the Federal Register.

3. DELIVERY AND ACCEPTANCE

3.1 Delivery Window
Hardware shall be delivered within 14 calendar days of a Purchase Order. The
Agency reserves the right to reject any delivery arriving after 30 calendar days
without prior written notice of delay.

3.2 Acceptance Testing
Delivered equipment undergoes a 5-business-day acceptance review. The Agency
shall notify the Vendor of any defects within this period. Silence constitutes
acceptance. Rejected items must be replaced within 10 business days at no charge.

4. PROHIBITION ON WEEKEND DELIVERIES AND INVOICES

4.1 Any delivery accepted outside of standard business hours (Monday–Friday,
07:00–19:00) must be pre-approved in writing and logged in the Contract Management
System. Weekend or holiday deliveries that appear without such approval shall be
flagged for audit review as potentially unauthorized expenditures.

5. CONFLICT OF INTEREST AND FRAUD PREVENTION

5.1 The Vendor warrants that no Agency employee or official has a personal
financial interest in this contract.
5.2 Vendor shall promptly report any attempt by an Agency employee to solicit
gifts, favors, or kickbacks.
5.3 Invoices that are round-number amounts ($10,000.00, $25,000.00, $100,000.00)
that do not correspond to itemized deliverables shall be treated as irregular and
subject to secondary review.

6. TERMINATION

6.1 For Convenience
The Agency may terminate this contract at any time with 30 days' written notice.
Vendor shall be compensated for authorized work completed prior to termination.

6.2 For Cause
Immediate termination is permitted if Vendor engages in fraud, misrepresentation,
or material breach of any clause of this Agreement.

7. GOVERNING LAW
This Agreement is governed by the Federal Acquisition Regulations (FAR) and the
laws of the United States of America.
""",
    },

    "buildright": {
        "title": "Facility Renovation Services Agreement – BuildRight Construction",
        "vendor_name": "BuildRight Construction Inc.",
        "text": """
FACILITY RENOVATION SERVICES AGREEMENT

CONTRACT NUMBER: FAC-2024-0044
EFFECTIVE DATE: March 1, 2024
EXPIRATION DATE: February 28, 2026
TOTAL CONTRACT VALUE: $5,750,000.00

PARTIES
This Agreement is between the Department of General Services ("Agency") and
BuildRight Construction Inc. ("Contractor"), a licensed general contractor
operating in the District of Columbia.

1. SCOPE OF WORK

1.1 Renovation Work
Contractor shall perform building renovation services as described in Attachment 1
(Statement of Work), including HVAC upgrades, electrical panel replacement,
ADA-compliance modifications, and exterior facade repair.

1.2 Project Management
Contractor shall assign a dedicated Project Manager who shall attend weekly
progress meetings with the Agency's Contracting Officer's Representative (COR).

2. PRICING AND PAYMENT

2.1 Milestone-Based Pricing
Payment is structured around six project milestones defined in Attachment 2.
Each milestone payment shall be triggered only upon written acceptance by the COR.
No advance payments are permitted.

2.2 Change Orders
Any work outside the original Statement of Work requires a written Change Order
approved by the Contracting Officer before work commences. Unauthorized work
performed without a Change Order shall not be reimbursed.

2.3 Maximum Invoice Frequency
Contractor shall submit progress invoices no more than bi-weekly (every two weeks).
Invoices submitted within 7 calendar days of a preceding invoice for the same
project phase shall be deemed duplicate and returned unpaid.

2.4 Subcontractor Disclosure
All subcontractors with individual task values exceeding $5,000 must be disclosed
to the Agency within 5 business days of engagement. Undisclosed subcontractors
may not be reimbursed.

3. PREVAILING WAGE REQUIREMENTS

3.1 All workers on this project are subject to Davis-Bacon Act wage requirements.
Certified payroll records must be submitted monthly. Failure to pay prevailing
wages is grounds for immediate contract suspension.

4. PROHIBITED BILLING PRACTICES

4.1 No billing for idle time, standby time, or mobilization costs unless
specifically authorized in writing.
4.2 Materials must be billed at cost plus a maximum 10% markup. The Agency
reserves the right to audit Contractor purchase receipts at any time.
4.3 Pattern of repeated invoices just below approval thresholds ($9,900–$9,999
or $24,800–$24,999) shall be treated as potential bid-splitting and referred
to the Office of Inspector General.

5. BONDING AND INSURANCE
Contractor shall maintain a performance bond equal to 100% of contract value
and a payment bond equal to 100% of contract value throughout the contract period.

6. DISPUTE RESOLUTION
Disputes shall first be mediated by the Agency's Board of Contract Appeals before
proceeding to federal court.
""",
    },

    "medsupply": {
        "title": "Medical Equipment and Supplies Agreement – MedSupply Corp",
        "vendor_name": "MedSupply Corp",
        "text": """
MEDICAL EQUIPMENT AND SUPPLIES PROCUREMENT AGREEMENT

CONTRACT NUMBER: MED-2024-0019
EFFECTIVE DATE: April 1, 2024
EXPIRATION DATE: March 31, 2026
TOTAL CONTRACT VALUE: $1,800,000.00

PARTIES
This Agreement is between the Department of Health Services ("Agency") and
MedSupply Corp ("Vendor"), an FDA-registered medical device distributor.

1. PRODUCTS AND SERVICES

1.1 Covered Products
Vendor shall supply Class II medical devices, disposable supplies, and
maintenance services as enumerated in Product Schedule C.

1.2 Regulated Items
All products must hold current FDA clearance or 510(k) approval.
Vendor shall notify the Agency within 24 hours if any product is subject
to a recall, market withdrawal, or safety alert.

2. PRICING

2.1 Catalog Pricing
Products are sold at catalog unit prices with the Agency receiving a
negotiated 15% discount from Vendor's published GSA schedule price.

2.2 Emergency Orders
Emergency procurement orders (defined as needed within 48 hours) may be
placed at standard catalog pricing without the 15% discount. Emergency
orders shall not exceed $50,000 per order without explicit prior approval
from the Medical Device Procurement Committee.

2.3 Quantity Thresholds and Split Order Prohibition
Orders for the same product category within any 30-day rolling window
shall not be split across multiple Purchase Orders to circumvent the
$10,000 simplified acquisition threshold. Violations will be immediately
escalated to the Procurement Compliance Officer.

2.4 Round-Number Billing Alert
Invoices with amounts that are exact round numbers ($5,000.00, $10,000.00,
$25,000.00) without corresponding itemized line items shall be flagged as
anomalous.  The Vendor must provide itemized receipts for all such invoices.

3. DELIVERY AND COLD CHAIN

3.1 Temperature-Sensitive Products
Products requiring cold-chain handling must be shipped with calibrated
temperature loggers. Deliveries failing cold-chain requirements shall be
rejected and replaced at Vendor's cost.

3.2 48-Hour Delivery SLA
Standard orders must be fulfilled within 48 hours of confirmed Purchase Order.
Late deliveries shall incur a 2% penalty per day, capped at 20% of order value.

4. AUDIT RIGHTS

4.1 The Agency reserves the right to audit Vendor's invoices, pricing records,
and delivery documentation for up to 5 years after contract expiration.
4.2 Any pattern of billing near but below approval thresholds shall trigger
an automatic compliance review.

5. FRAUD AND BILLING CONTROLS
Vendor acknowledges that the Agency uses automated anomaly detection tools
and that suspicious billing patterns including: rapid sequential invoices,
round-number invoices, weekend submissions, and amounts just below approval
thresholds, will be automatically escalated for human review.

6. TERMINATION AND REMEDIES
Material breach, including fraudulent invoicing, results in immediate contract
termination, repayment of all amounts paid, and referral to the Department
of Justice for False Claims Act prosecution.
""",
    },

    "logitrans": {
        "title": "Logistics and Transportation Services – LogiTrans Freight",
        "vendor_name": "LogiTrans Freight LLC",
        "text": """
LOGISTICS AND TRANSPORTATION SERVICES AGREEMENT

CONTRACT NUMBER: LOG-2024-0057
EFFECTIVE DATE: February 15, 2024
EXPIRATION DATE: February 14, 2026
TOTAL CONTRACT VALUE: $980,000.00

PARTIES
This Agreement is between the Supply Chain Management Office ("Agency") and
LogiTrans Freight LLC ("Carrier"), a DOT-licensed motor carrier.

1. TRANSPORTATION SERVICES

1.1 LTL and FTL Shipments
Carrier shall provide Less-Than-Truckload (LTL) and Full-Truckload (FTL)
freight services for Agency equipment and materials throughout the contiguous
United States.

1.2 Last-Mile Delivery
Carrier shall provide last-mile white-glove delivery for fragile or sensitive
equipment.

2. RATE SCHEDULE

2.1 Per-Mile Rates
Freight rates are specified in Rate Schedule D by shipment class (Class 50–500).
Rates are firm-fixed for the contract period and may not be unilaterally adjusted.

2.2 Fuel Surcharge Cap
Fuel surcharges are capped at 8% of the base freight rate regardless of market
fuel prices. Any surcharge above the cap requires a formal contract modification.

2.3 Billing Frequency and Consolidation
Carrier shall submit invoices weekly consolidating all prior-week shipments
on a single Bill of Lading summary. Daily invoicing is not permitted and
submissions more frequent than weekly shall be returned unpaid without penalty.

2.4 No Split Billing on Single Consignments
A single consignment confirmed in one Bill of Lading may not be split across
multiple invoices. Splitting a single consignment to reduce apparent individual
invoice amounts is a material breach of this Agreement.

3. PERFORMANCE STANDARDS

3.1 On-Time Delivery
Carrier guarantees 95% on-time delivery. If monthly on-time rate falls below
90%, the Agency may levy a 5% credit against the following month's invoices.

3.2 Damage Claims
Carrier liability for cargo damage is limited to $0.50 per pound per package,
unless higher declared value is purchased.

4. PROHIBITED PRACTICES

4.1 No Re-brokering of Loads
Carrier may not re-broker Agency shipments to third-party carriers without
explicit written consent. Violation voids Carrier's limited liability protection
and transfers full risk to Carrier.

4.2 Round-Trip Billing Integrity
Carrier shall not invoice for return trips unless a return load was confirmed
by the Agency in writing.

5. AUDIT AND RECORD RETENTION
All waybills, Bills of Lading, and GPS delivery records must be retained for
3 years and made available within 5 business days of an audit request.

6. FORCE MAJEURE
Carrier is excused from performance during documented weather events, natural
disasters, or government-declared emergencies, provided notice is given within
24 hours of the impeding event.
""",
    },

    "cloudscale": {
        "title": "Professional Consulting Services Agreement – CloudScale Consulting",
        "vendor_name": "CloudScale Consulting Partners",
        "text": """
PROFESSIONAL CONSULTING SERVICES AGREEMENT

CONTRACT NUMBER: CON-2024-0033
EFFECTIVE DATE: January 15, 2024
EXPIRATION DATE: January 14, 2026
TOTAL CONTRACT VALUE: $3,200,000.00

PARTIES
This Agreement is between the Office of Digital Transformation ("Agency") and
CloudScale Consulting Partners ("Consultant"), a management consulting firm.

1. SERVICES

1.1 Digital Transformation Advisory
Consultant shall provide strategic advisory services for the Agency's cloud-first
modernization program, including cloud architecture design, procurement strategy,
change management, and workforce training.

1.2 Authorized Personnel
Services shall be delivered by the named key personnel listed in Exhibit B.
Substitution of key personnel requires 30 days' prior notice and Agency approval.
Any unapproved substitution allows the Agency to withhold payment.

2. COMPENSATION

2.1 Labor Category Rates
Consultant's billable rates by labor category are:
  - Principal Consultant:    $295/hour
  - Senior Consultant:       $225/hour
  - Associate Consultant:    $165/hour
  - Analyst:                 $125/hour

These rates are inclusive of all overhead, profit, and general expenses.
No separate invoicing for travel, communications, or facilities is permitted
unless pre-approved in writing via a Travel Authorization Form.

2.2 Month-End Billing
Consultant shall submit a single consolidated invoice per calendar month
accompanied by detailed timesheets for all personnel. Invoices for partial
months or for single engagements spread across two invoices in one month
shall be rejected.

2.3 Not-to-Exceed Task Orders
Individual task orders are issued with Not-To-Exceed (NTE) amounts. Consultant
may not exceed an NTE by more than 10% without prior written modification.
Overruns are not reimbursable without a signed contract modification.

2.4 Approval Thresholds for Task Orders
Task orders with a value of $25,000 – $99,999 require Deputy Contracting
Officer approval. Task orders from $100,000 – $249,999 require Chief
Procurement Officer approval. Task orders at or above $250,000 require
full contract modification with Agency Executive sign-off.

3. CONFLICT OF INTEREST

3.1 Organizational Conflict of Interest (OCI)
Consultant represents that it has no organizational or financial conflict of
interest with Agency programs being evaluated under this contract.

3.2 Non-Solicitation of Agency Employees
Consultant shall not recruit, solicit, or offer employment to any Agency employee
involved in the procurement or oversight of this contract for one year post-contract.

4. DELIVERABLES AND ACCEPTANCE

4.1 Written Deliverables
All written reports must be delivered in editable formats. A deliverable is
considered accepted when the Agency COR signs the Deliverable Acceptance Form.
The COR has 10 business days after delivery to accept, reject, or request revisions.

4.2 Repeated Invoice Velocity Alert
If Consultant submits more than two invoices in a single calendar month or
invoices totaling more than 15% above their average monthly billing over the
prior 90 days, the Agency's automated billing audit system shall flag the invoice
for secondary review before authorizing payment.

5. INTELLECTUAL PROPERTY
All deliverables produced under this contract are works made for hire and
become the exclusive property of the Agency upon acceptance.

6. DATA SECURITY AND CONFIDENTIALITY
Consultant shall comply with FISMA Moderate controls for any Agency data
accessed during performance.  Data must be encrypted at rest (AES-256) and
in transit (TLS 1.2+).

7. GOVERNING LAW AND DISPUTES
This Agreement is governed by federal procurement law. Disputes shall be
resolved per the Contract Disputes Act, 41 U.S.C. §§ 7101-7109.
""",
    },
}
