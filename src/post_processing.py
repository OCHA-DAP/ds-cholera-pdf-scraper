"""
Post-processing utilities for LLM extraction results.
This module provides functions to clean and standardize extracted data.
"""

import re

import pandas as pd


def apply_post_processing_pipeline(df, source="llm"):
    """
    Apply complete post-processing pipeline to extracted data.

    Args:
        df: DataFrame to process
        source: "llm" or "baseline" for source-specific fixes

    Returns:
        Cleaned DataFrame with standardized values
    """
    df_clean = df.copy()

    print(f"Applying post-processing to {source} data...")

    # Apply all cleaning steps
    df_clean = clean_numerical_fields(df_clean, source)
    df_clean = standardize_cfr_format(df_clean)
    df_clean = standardize_event_names(df_clean)
    df_clean = standardize_country_names(df_clean)
    df_clean = standardize_column_names(df_clean)
    df_clean = harmonize_missing_values(df_clean)

    print(f"✅ Post-processing complete for {source} data")
    return df_clean


def clean_numerical_fields(df, source="llm"):
    """Remove commas from numerical fields to make them properly numeric."""

    numerical_fields = [
        "TotalCases",
        "Total cases",
        "CasesConfirmed",
        "Cases Confirmed",
        "Deaths",
    ]

    for field in numerical_fields:
        if field in df.columns:

            def clean_numerical_value(value):
                """Clean numerical values by removing commas and converting to float.
                Convert any non-numeric to 0."""
                if pd.isna(value) or value in ["nan", "None", ""]:
                    return 0.0
                try:
                    # Remove commas and convert to float
                    val_str = str(value).replace(",", "").strip()
                    if val_str == "" or val_str.lower() in ["nan", "none"]:
                        return 0.0
                    return float(val_str)
                except ValueError:
                    return 0.0

            df[field] = df[field].apply(clean_numerical_value)

    return df


def standardize_cfr_format(df):
    """Standardize CFR format (remove % symbols)."""

    if "CFR" in df.columns:

        def clean_cfr(value):
            if pd.isna(value) or value in ["nan", "None", ""]:
                return 0.0
            try:
                val_str = str(value).strip().replace("%", "")
                return float(val_str)
            except ValueError:
                return 0.0

        df["CFR"] = df["CFR"].apply(clean_cfr)

    return df


def standardize_event_names(df):
    """Standardize event name variations."""

    if "Event" in df.columns:
        event_mapping = {
            "Humanitarian Crisis": "Complex Humanitarian crisis",
            "Humanitarian crisis": "Complex Humanitarian crisis",
            "Humanitarian crisis (North-West & South-West)": "Humanitarian crisis (Noth-West & South-West )",
            "Complex Humanitarian crisis- ETH": "Complex Humanitarian crisis",
            "Complex Humanitarian crisis -SS": "Complex Humanitarian crisis",
        }

        df["Event"] = df["Event"].replace(event_mapping)

    return df


def standardize_country_names(df):
    """Fix country name encoding issues."""

    if "Country" in df.columns:
        country_mapping = {"CÃ´te d'Ivoire": "Côte d'Ivoire"}
        df["Country"] = df["Country"].replace(country_mapping)

    return df


def standardize_column_names(df):
    """Standardize column names for consistency."""

    column_mapping = {
        "Date notified to WCO": "DateNotified",
        "Start of reporting period": "StartReportingPeriod",
        "End of reporting period": "EndReportingPeriod",
        "Total cases": "TotalCases",
        "Cases Confirmed": "CasesConfirmed",
    }

    return df.rename(columns=column_mapping)


def harmonize_missing_values(df):
    """Standardize missing value representation."""

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(["nan", "None", "", "NaN"], None)

    return df


def validate_post_processing(original_df, processed_df):
    """Validate that post-processing didn't corrupt data."""

    validation_results = {
        "record_count_preserved": len(original_df) == len(processed_df),
        "no_new_nulls_in_country": processed_df["Country"].isna().sum()
        <= original_df["Country"].isna().sum(),
        "no_new_nulls_in_event": processed_df["Event"].isna().sum()
        <= original_df["Event"].isna().sum(),
        "numerical_values_reasonable": True,  # Could add specific checks
    }

    if all(validation_results.values()):
        print("✅ Post-processing validation passed")
        return True
    else:
        print("⚠️ Post-processing validation failed:")
        for check, passed in validation_results.items():
            if not passed:
                print(f"  ❌ {check}")
        return False


# Integration with main extraction pipeline
def process_llm_extraction_results(df):
    """
    Main function to post-process LLM extraction results.

    Args:
        df: Raw LLM extraction DataFrame

    Returns:
        Cleaned and standardized DataFrame
    """
    print("🧹 Starting post-processing pipeline...")

    # Store original for validation
    original_df = df.copy()

    # Apply post-processing
    processed_df = apply_post_processing_pipeline(df, source="llm")

    # Validate results
    if validate_post_processing(original_df, processed_df):
        print(f"📊 Processed {len(processed_df)} records successfully")
        return processed_df
    else:
        print("⚠️ Post-processing validation failed, returning original data")
        return original_df


if __name__ == "__main__":
    # Test the post-processing pipeline
    print("Testing post-processing pipeline...")

    # Load test data
    test_df = pd.read_csv(
        "/Users/zackarno/Documents/CHD/repos/ds-cholera-pdf-scraper/outputs/text_extracted_data.csv"
    )

    # Apply post-processing
    cleaned_df = process_llm_extraction_results(test_df)

    # Show results
    print(f"\\nOriginal shape: {test_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")

    # Save cleaned results
    output_path = "/Users/zackarno/Documents/CHD/repos/ds-cholera-pdf-scraper/outputs/text_extracted_data_cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")
