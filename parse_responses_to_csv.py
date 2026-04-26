import json
import csv
import re
import argparse
import os

def parse_responses(input_json="response.json", output_csv="structured_responses.csv"):
    if not os.path.exists(input_json):
        print(f"❌ Error: Cannot find '{input_json}'.")
        return

    # Load the JSON data
    with open(input_json, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return

    if not data:
        print("Dataset is empty.")
        return

    print(f"Loaded {len(data)} records from {input_json}.")

    structured_data = []

    for idx, row in enumerate(data):
        file_name = row.get("file_name", f"unknown_{idx}")
        prompt = row.get("prompt", "")
        raw_response = row.get("response", "")

        verdict, reasoning = "", ""
        obs, asc, conc = "", "", ""

        # 1. Try New Structure: [Verdict] and [Reasoning]
        # Regex is flexible for [Verdict], [Verdict:], [Verdict]: etc.
        v_match = re.search(r'\[Verdict\]?:\s*(.*?)(?=\[Reasoning\][:]?|$)', raw_response, re.DOTALL | re.IGNORECASE)
        r_match = re.search(r'\[Reasoning\][:]?\s*(.*)', raw_response, re.DOTALL | re.IGNORECASE)
        
        if v_match: verdict = v_match.group(1).strip()
        if r_match: reasoning = r_match.group(1).strip()

        # 2. Try Old Structure (Fallback): [Observation], [Assessment], [Conclusion]
        obs_match = re.search(r'\[Observation\][:]?\s*(.*?)(?=\[Assessment\][:]?|$)', raw_response, re.DOTALL | re.IGNORECASE)
        asc_match = re.search(r'\[Assessment\][:]?\s*(.*?)(?=\[Conclusion\][:]?|$)', raw_response, re.DOTALL | re.IGNORECASE)
        conc_match = re.search(r'\[Conclusion\][:]?\s*(.*)', raw_response, re.DOTALL | re.IGNORECASE)

        if obs_match: obs = obs_match.group(1).strip()
        if asc_match: asc = asc_match.group(1).strip()
        if conc_match: conc = conc_match.group(1).strip()

        # If everything failed to format, use raw response in reasoning
        if not any([verdict, reasoning, obs, asc, conc]):
            reasoning = raw_response

        structured_data.append({
            "file_name": file_name,
            "prompt": prompt,
            "verdict": verdict,
            "reasoning": reasoning,
            "observation_old": obs,
            "assessment_old": asc,
            "conclusion_old": conc,
            "raw_response": raw_response
        })

    # Write out to CSV securely
    fieldnames = ["file_name", "prompt", "verdict", "reasoning", "observation_old", "assessment_old", "conclusion_old", "raw_response"]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(structured_data)

    print(f"\n✅ Successfully exported {len(structured_data)} structured responses to '{output_csv}'!")

if __name__ == "__main__":
    print("-" * 50)
    print("Structured Output Parser")
    print("-" * 50)
    
    input_file = input("\nEnter the JSON file to read from [default: response.json]: ").strip()
    if not input_file: 
        input_file = "response.json"
        
    output_csv = input("Enter the file name to save as (e.g. results.csv) [default: structured_responses.csv]: ").strip()
    if not output_csv: 
        output_csv = "structured_responses.csv"
        
    if not output_csv.lower().endswith(".csv"):
        output_csv += ".csv"
        
    print("-" * 50)
    parse_responses(input_file, output_csv)
