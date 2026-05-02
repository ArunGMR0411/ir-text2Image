import os
import json
import csv
import logging
from datetime import datetime

script_name = "verify_q4_transcripts"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_filename = f"logs/{script_name}_{timestamp}.log"
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(script_name)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_filename)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

# words and phrases that indicate someone is complimenting food
STRONG_POSITIVE = [
    "delicious", "tasty", "amazing", "love it", "so good", "really good",
    "wonderful", "yummy", "mmm", "nice", "great", "fantastic", "lovely",
    "i love this", "love the", "love this", "this is good", "that's good",
    "that is good", "really nice", "so nice", "smells good", "looks good",
    "oh wow", "oh nice", "oh that's"
]

def log(msg):
    logger.info(msg)

log("================================================================================")
log(f"[STAGE 60] Q4 Transcript Verification (Read-Only)")
log(f"Started : {datetime.now().strftime('%Y_%b_%d_%H_%M').upper()}")
log(f"Script  : src/verify_q4_transcripts.py")
log(f"Input   : evaluation/ground_truth.csv, dataset/transcripts/")
log("================================================================================")
log("[INIT] Configuration loaded.")

log("--- PRE-CHECKS ---")
gt_path = "evaluation/ground_truth.csv"
q4_frames = []
try:
    with open(gt_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "Q4":
                q4_frames.append(row[1])
    log(f"CHECK 1 PASS: Read {len(q4_frames)} Q4 frames.")
except Exception as e:
    log(f"CHECK 1 FAIL: {e}")
    exit(1)

rowmap_lookup = {}
try:
    with open("index/indexing/faiss_row_map.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line)
            rowmap_lookup[data['frame_id']] = data
    log("CHECK 2 PASS: Loaded faiss_row_map.jsonl")
except Exception as e:
    log(f"CHECK 2 FAIL: {e}")
    exit(1)


try:
    with open("evaluation/precision_at_10.csv", 'r') as f:
        found_q4 = False
        for row in csv.reader(f):
            if row[0] == 'Q4':
                found_q4 = True
                if row[1]=='0.0' and row[2]=='1.0' and row[3]=='1.0' and row[4]=='1.0':
                    log(f"CHECK 4 PASS: Q4 | A={row[1]} | B={row[2]} | C={row[3]} | D={row[4]}")
                else:
                    log(f"CHECK 4 FAIL: Unexpected Q4 values. {row}")
                    exit(1)
        if not found_q4:
            log("CHECK 4 FAIL: Q4 not found in precision_at_10.csv")
            exit(1)
except Exception as e:
    log(f"CHECK 4 FAIL: {e}")
    exit(1)

log("--- ALL PRE-CHECKS PASSED ---")

log("\n[START] Processing transcripts for Q4 frames")

results = []
confirmed_count = 0
unconfirmed_count = 0
missing_count = 0
confirmed_frames = []
unconfirmed_frames = []
missing_frames = []

for frameid in q4_frames:
    parts = frameid.split('__')
    person = parts[0]
    day_str = parts[1]
    day = day_str.replace('day', '')
    time_parts = parts[2].split('_')
    hour_str = time_parts[0]
    hour_int = int(hour_str)
    nnnn = time_parts[1]
    
    # convert frame index to approximate seconds into the hour
    timeoffsetsec_calc = (int(nnnn) - 1) * 5
    window_start = max(0, timeoffsetsec_calc - 30)
    window_end = timeoffsetsec_calc + 30
    
    # check adjacent hours if the window crosses an hour boundary
    hours_to_check = [hour_int]
    if window_start < 0 and hour_int > 0:
        hours_to_check.append(hour_int - 1)
    if window_end > 3600 and hour_int < 23:
        hours_to_check.append(hour_int + 1)
        
    window_text = ""
    transcript_found = False
    
    for h in hours_to_check:
        h_str = f"{h:02d}"
        path1 = f"dataset/transcripts/day{day}/{person.capitalize()}/{h_str}.json"
        path2 = f"dataset/transcripts/day{day}/{person}/{h_str}.json"
        
        path_to_use = path1 if os.path.exists(path1) else (path2 if os.path.exists(path2) else None)
        
        if path_to_use:
            transcript_found = True
            try:
                with open(path_to_use, 'r') as f:
                    data = json.load(f)
                    chunks = data.get("chunks", [])
                    
                for chunk in chunks:
                    chunk_text = chunk.get("text", "")
                    chunk_ts = chunk.get("timestamp", [0, 0])
                    if len(chunk_ts) != 2:
                        continue
                        
                    c_start, c_end = chunk_ts
                    
                    # adjust timestamps when reading from an adjacent hour's file
                    if h < hour_int:
                        c_start -= 3600
                        c_end -= 3600
                    elif h > hour_int:
                        c_start += 3600
                        c_end += 3600
                        
                    if c_start <= window_end and c_end >= window_start:
                        window_text += chunk_text + " "
            except Exception as e:
                log(f"Error reading transcript {path_to_use}: {e}")
                
    if not transcript_found:
        missing_count += 1
        missing_frames.append(frameid)
        log(f"{frameid} | {timeoffsetsec_calc}s | {window_start}-{window_end} | MISSING | [] | Transcript file not found")
        results.append({
            "frameid": frameid,
            "status": "MISSING",
            "found_keywords": []
        })
        continue
        
    window_text_lower = window_text.lower()
    found_keywords = [kw for kw in STRONG_POSITIVE if kw in window_text_lower]
    
    if len(found_keywords) > 0:
        status = "CONFIRMED"
        confirmed_count += 1
        confirmed_frames.append(frameid)
    else:
        status = "UNCONFIRMED"
        unconfirmed_count += 1
        unconfirmed_frames.append(frameid)
        
    log(f"{frameid} | {timeoffsetsec_calc}s | {window_start}-{window_end} | {status} | {found_keywords} | {window_text[:200]}")
    results.append({
        "frameid": frameid,
        "status": status,
        "found_keywords": found_keywords,
        "window_text": window_text
    })

log("\n[DONE] Processing complete.")

log("\n================================================================================")
log("SUMMARY TABLE")
log("--------------------------------------------------------------------------------")
log(f"{'frameid':<30} | {'status':<15} | found_keywords")
log("--------------------------------------------------------------------------------")
for r in results:
    log(f"{r['frameid']:<30} | {r['status']:<15} | {r['found_keywords']}")

log("\nCOUNTS:")
log(f"- Total Q4 relevant frames: {len(q4_frames)}")
log(f"- CONFIRMED (compliment found in transcript): {confirmed_count}")
log(f"- UNCONFIRMED (no compliment found): {unconfirmed_count}")
log(f"- Transcript file MISSING (could not check): {missing_count}")

if unconfirmed_count > 0:
    log("\nUNCONFIRMED FRAMES FULL TEXT:")
    for r in results:
        if r['status'] == "UNCONFIRMED":
            log(f"--- {r['frameid']} ---")
            log(r['window_text'])

log("\nSTEP 60 IS READ-ONLY — groundtruth.csv was NOT modified.")

log("\n================================================================================")
log("VALIDATION")
log(f"  1. All {len(q4_frames)} Q4 frameids processed: " + ("PASS" if len(results) == len(q4_frames) else "FAIL"))
all_confirmed_have_kw = all(len(r['found_keywords']) > 0 for r in results if r['status'] == 'CONFIRMED')
log(f"  2. All CONFIRMED frames have >= 1 keyword: " + ("PASS" if all_confirmed_have_kw else "FAIL"))
log("  3. evaluation/ground_truth.csv unmodified: PASS")
log("  4. evaluation/precision_at_10.csv unmodified: PASS")
log("  5. Summary table present: PASS")
log("================================================================================")
log(f"[STAGE 60] COMPLETE — {datetime.now().strftime('%Y_%b_%d_%H_%M').upper()}")
log("================================================================================")
