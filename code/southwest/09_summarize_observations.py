from _support import stage_cli

if __name__ == "__main__":
    stage_cli("09_observation_statistics", (
        "09_1_build_lake_observation_status_records_result3.py",
        "09_2_build_lake_observation_summary_result3.py",
        "09_3_build_lake_observation_resummary_result3.py",
    ))
