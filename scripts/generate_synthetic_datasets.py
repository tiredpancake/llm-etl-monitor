"""
Generates the two remaining evaluation datasets described in the proposal
(Section 9) that do not have a single canonical downloadable file:

- openml_dirty        -> "OpenML Dirty Tabular" (Schema Mismatch, Semantic Error)
- synthetic_llmclean   -> synthetic LLMClean-style dataset (Semantic Errors, Mixed Formats)

OpenML's "dirty tabular" datasets are a *category* of many different
community-uploaded tables, not one fixed file, and the proposal's own
"synthetic (LLMClean)" entry is explicitly generated rather than downloaded.
So both are built here as reproducible, seeded synthetic generators that
inject the *exact error types the proposal lists* for each dataset, which is
more useful for testing the pipeline's routing logic than any single
real-world file would be.

Run:
    python scripts/generate_synthetic_datasets.py
"""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ---------------------------------------------------------------------------
# Dataset 1: openml_dirty  (~2000 rows) -> Schema Mismatch, Semantic Error
# ---------------------------------------------------------------------------
def generate_openml_dirty(n=2000) -> pd.DataFrame:
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Support"]
    countries_clean = ["USA", "Germany", "France", "Canada", "India", "Brazil"]
    country_variants = {
        "USA": ["USA", "United States", "U.S.A", "US", "united states"],
        "Germany": ["Germany", "Deutschland", "GER", "germany"],
        "France": ["France", "FR", "france"],
        "Canada": ["Canada", "CAN", "canada"],
        "India": ["India", "IN", "india"],
        "Brazil": ["Brazil", "Brasil", "BR"],
    }
    first_names = ["John", "Maria", "Wei", "Fatima", "Carlos", "Anna", "Kenji", "Olga", "Liam", "Priya"]
    last_names = ["Smith", "Garcia", "Chen", "Khan", "Silva", "Muller", "Tanaka", "Ivanov", "Brown", "Patel"]

    rows = []
    today = datetime(2026, 1, 1)
    for i in range(n):
        emp_id = 100000 + i
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        dept = random.choice(departments)
        country = random.choice(countries_clean)

        # --- schema mismatch: age sometimes numeric, sometimes spelled out ---
        age_val = random.randint(19, 68)
        if random.random() < 0.06:
            age_repr = {25: "twenty-five", 30: "thirty", 40: "forty"}.get(age_val, str(age_val))
        else:
            age_repr = age_val
        # semantic error: a few impossible ages
        if random.random() < 0.02:
            age_repr = random.choice([-5, 187, 0])

        # --- schema mismatch: salary as number, currency string, or with text unit ---
        base_salary = random.randint(35000, 160000)
        r = random.random()
        if r < 0.15:
            salary_repr = f"${base_salary}"
        elif r < 0.25:
            salary_repr = f"{base_salary} USD"
        else:
            salary_repr = base_salary

        # --- schema mismatch: boolean column encoded inconsistently ---
        is_mgr = random.random() < 0.2
        is_mgr_repr = random.choice(
            [str(is_mgr), "yes" if is_mgr else "no", 1 if is_mgr else 0, "TRUE" if is_mgr else "FALSE"]
        )

        # --- semantic error: hire_date sometimes in the future / malformed order ---
        days_ago = random.randint(30, 6000)
        hire_date = today - timedelta(days=days_ago)
        if random.random() < 0.02:
            hire_date = today + timedelta(days=random.randint(10, 400))  # impossible future hire
        hire_date_repr = hire_date.strftime(random.choice(["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"]))

        # --- schema mismatch: country in inconsistent forms ---
        country_repr = random.choice(country_variants[country])

        # --- semantic error: email not matching name pattern, or malformed ---
        email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@company.com"
        if random.random() < 0.03:
            email = name.split()[0].lower() + "AT company dot com"  # malformed

        # --- semantic error: performance score outside its valid 1-10 scale ---
        score = random.randint(1, 10)
        if random.random() < 0.02:
            score = random.choice([-3, 15, 99])

        rows.append({
            "employee_id": emp_id,
            "full_name": name,
            "age": age_repr,
            "department": dept,
            "salary": salary_repr,
            "hire_date": hire_date_repr,
            "email": email,
            "country": country_repr,
            "is_manager": is_mgr_repr,
            "performance_score": score,
        })

    df = pd.DataFrame(rows)

    # inject a handful of exact duplicate rows (Duplicate error type used elsewhere too)
    dup_sample = df.sample(n=max(1, n // 100), random_state=SEED)
    df = pd.concat([df, dup_sample], ignore_index=True)

    # inject some missing values across a few columns (schema/semantic layer still needs
    # baseline completeness gaps for the profiler to pick up)
    for col in ["salary", "country", "performance_score"]:
        idx = df.sample(frac=0.03, random_state=SEED).index
        df.loc[idx, col] = np.nan

    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset 2: synthetic_llmclean (~1000 rows, 10 cols) -> Semantic Error, Mixed Format
# ---------------------------------------------------------------------------
def generate_synthetic_llmclean(n=1000) -> pd.DataFrame:
    categories = ["Electronics", "Home & Kitchen", "Books", "Toys", "Clothing", "Sports"]
    countries = ["United States", "Germany", "United Kingdom", "France", "Japan", "Australia"]
    payment_methods = ["credit_card", "paypal", "bank_transfer", "cash_on_delivery"]
    statuses = ["placed", "shipped", "delivered", "cancelled", "returned"]
    first_names = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie"]
    last_names = ["Lee", "Kim", "Nguyen", "Costa", "Weber", "Rossi", "Dubois", "Novak"]

    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"]
    rows = []
    base_date = datetime(2025, 1, 1)

    for i in range(n):
        order_id = f"ORD-{10000 + i}"
        customer = f"{random.choice(first_names)} {random.choice(last_names)}"
        order_dt = base_date + timedelta(days=random.randint(0, 500))
        order_date_repr = order_dt.strftime(random.choice(date_formats))

        category = random.choice(categories)
        price = round(random.uniform(5, 500), 2)
        r = random.random()
        if r < 0.2:
            price_repr = f"${price}"
        elif r < 0.35:
            price_repr = f"{price} USD"
        else:
            price_repr = price
        # semantic error: negative price
        if random.random() < 0.02:
            price_repr = -abs(price)

        quantity = random.randint(1, 5)
        status = random.choice(statuses)
        # semantic error: cancelled/returned orders with delivered-only fields, or
        # quantity 0 combined with a "delivered" status (logically inconsistent)
        if random.random() < 0.03:
            quantity = 0
            status = "delivered"

        country = random.choice(countries)
        payment = random.choice(payment_methods)

        # mixed format phone numbers
        phone_variants = [
            f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}",
            f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
            f"{random.randint(1000000000,9999999999)}",
        ]
        phone = random.choice(phone_variants)
        # semantic error: malformed / impossible phone number
        if random.random() < 0.02:
            phone = "N/A"

        rows.append({
            "order_id": order_id,
            "customer_name": customer,
            "order_date": order_date_repr,
            "product_category": category,
            "price": price_repr,
            "quantity": quantity,
            "shipping_country": country,
            "payment_method": payment,
            "order_status": status,
            "phone_number": phone,
        })

    df = pd.DataFrame(rows)

    # a few missing values scattered (mostly in optional-looking fields)
    for col in ["phone_number", "payment_method"]:
        idx = df.sample(frac=0.02, random_state=SEED).index
        df.loc[idx, col] = np.nan

    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dirty = generate_openml_dirty()
    dirty_path = os.path.join(OUT_DIR, "openml_dirty.csv")
    dirty.to_csv(dirty_path, index=False)
    print(f"openml_dirty.csv       -> {dirty.shape[0]} rows x {dirty.shape[1]} cols  ({dirty_path})")

    synth = generate_synthetic_llmclean()
    synth_path = os.path.join(OUT_DIR, "synthetic_llmclean.csv")
    synth.to_csv(synth_path, index=False)
    print(f"synthetic_llmclean.csv -> {synth.shape[0]} rows x {synth.shape[1]} cols  ({synth_path})")


if __name__ == "__main__":
    main()
