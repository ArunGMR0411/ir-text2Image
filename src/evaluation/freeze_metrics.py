import subprocess
import sys
import datetime
import os

def log(message):
    sys.stdout.flush()

def main():
    log("================================================================================")
    log("[STAGE 63] Final P@10 Freeze")
    log(f"Started : {datetime.datetime.now().strftime('%Y_%b_%d_%H_%M').upper()}")
    log("Script  : src/freeze_metrics.py")
    log("Input   : src/calculate_metrics.py")
    log("================================================================================")
    log("[INIT] Configuration loaded.")
    log("[START] Re-running metrics from scratch")

    try:
        result = subprocess.run(
            ["venv/bin/python", "src/calculate_metrics.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        log("--- STDOUT ---")
        log(result.stdout)
        
        log("--- STDERR ---")
        log(result.stderr)
        
        if result.returncode != 0:
            log(f"FAIL: calculate_metrics.py returned exit code {result.returncode}")
            sys.exit(1)
            
        log("[DONE] Metrics re-calculated successfully.")
        
        log("================================================================================")
        log("VALIDATION")
        log("  1. calculate_metrics.py executed successfully: PASS")
        log("================================================================================")
        log(f"[STAGE 63] COMPLETE — {datetime.datetime.now().strftime('%Y_%b_%d_%H_%M').upper()}")
        log("================================================================================")
        
    except subprocess.TimeoutExpired:
        log("FAIL: calculate_metrics.py timed out after 120 seconds")
        sys.exit(1)
    except Exception as e:
        log(f"FAIL: Exception running calculate_metrics.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
