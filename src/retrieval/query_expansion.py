import datetime
import logging
import os
import sys

# synonyms, location hints, and object hints used to expand each query before retrieval
QUERY_ENRICHMENT = {
    "Q1": {
        "synonyms": ["hand blender", "electric whisk", "toaster", "hand mixer", "electric gadget", "plug", "buzzing", "portable appliance", "power cord", "on", "small appliance"],
        "location_hint": ["kitchen", "fixed/kitchen"],
        "color_hint": [],
        "object_hint": ["power cord", "plug", "portable device", "electric tool"]
    },
    "Q2": {
        "synonyms": ["espresso", "latte", "cappuccino", "brew", "coffee maker", "espresso machine", "cup", "press", "pod", "grounds", "coffee machine", "making coffee"],
        "location_hint": ["kitchen", "fixed/kitchen"],
        "color_hint": [],
        "object_hint": ["coffee machine body", "cup placement", "hand on controls"]
    },
    "Q3": {
        "synonyms": ["chocolate", "candy", "sweet", "snack", "wrapper", "eat", "bite", "unwrap", "munch", "foil", "plastic wrapper", "tearing", "crinkling", "biting into sweet"],
        "location_hint": ["any"],
        "color_hint": [],
        "object_hint": ["foil wrapper", "candy wrapper", "snack item"]
    },
    "Q4": {
        "synonyms": ["delicious", "tasty", "amazing", "love it", "good", "great", "wonderful", "yummy", "mmm", "nice", "compliment", "praise food", "I love this", "so good"],
        "location_hint": ["kitchen", "living1", "living2", "meeting"],
        "color_hint": [],
        "object_hint": ["food", "meal", "dish", "plate"]
    },
    "Q5": {
        "synonyms": ["singing", "song", "melody", "la la", "oh oh", "na na", "lyrics", "chorus", "vocal", "chant", "group singing"],
        "location_hint": ["any"],
        "color_hint": [],
        "object_hint": ["open mouth", "multiple people", "vocal expression"]
    },
    "Q6": {
        "synonyms": ["yellow octopus", "rubber octopus", "plush octopus", "octopus toy", "yellow toy", "tentacles", "yellow rubber", "small toy"],
        "location_hint": ["any"],
        "color_hint": ["yellow"],
        "object_hint": ["round body", "tentacles", "yellow hue", "small figurine"]
    },
    "Q7": {
        "synonyms": ["squirrel ornament", "christmas ornament", "squirrel figurine", "tree decoration", "christmas tree", "decorative squirrel", "holiday ornament", "squirrel christmas"],
        "location_hint": ["living1", "living2"],
        "color_hint": [],
        "object_hint": ["christmas tree", "ornament shape", "squirrel figurine", "decorative object on tree"]
    },
    "Q8": {
        "synonyms": ["cookie cutter", "bird cutter", "baking", "dough", "bird shape", "metal cutter", "bird silhouette", "baking supplies", "bird shaped metal", "cut out"],
        "location_hint": ["kitchen", "fixed/kitchen"],
        "color_hint": [],
        "object_hint": ["flat metal object", "bird outline", "plastic cutter", "baking tool"]
    },
    "Q9": {
        "synonyms": ["ace of spades", "playing card", "spade symbol", "card game", "deck of cards", "A spade", "cards", "card face", "spade pip"],
        "location_hint": ["meeting", "living1", "living2"],
        "color_hint": ["black"],
        "object_hint": ["playing card", "spade symbol", "A letter on card"]
    },
    "Q10": {
        "synonyms": ["bitten apple", "apple with bite", "partially eaten", "half apple", "apple bite marks", "apple missing flesh", "eaten fruit", "apple", "fruit", "bite mark"],
        "location_hint": ["any"],
        "color_hint": ["red", "green"],
        "object_hint": ["irregular apple edge", "bite region", "round fruit", "missing flesh"]
    }
}

RAW_QUERIES = {
    "Q1": "Using a portable electric kitchen gadget",
    "Q2": "Operating the coffee machine",
    "Q3": "Unwrapping a sweet snack and eating it",
    "Q4": "Somebody complimenting the food",
    "Q5": "People singing",
    "Q6": "Small yellow octopus toy",
    "Q7": "Squirrel christmas tree ornament",
    "Q8": "Bird-shaped cookie cutter",
    "Q9": "Ace of spades",
    "Q10": "Partially eaten apple"
}

def expand_query(query_id: str, raw_query_str: str) -> str:
    if query_id not in QUERY_ENRICHMENT:
        return raw_query_str.strip()
    
    entry = QUERY_ENRICHMENT[query_id]
    
    # concatenate all enrichment fields into one string for the encoder
    components = [raw_query_str] + entry["synonyms"] + entry["color_hint"] + entry["object_hint"]
    full_str = " ".join(components)
    
    tokens = full_str.split()
    seen = set()
    deduped = []
    
    # deduplicate tokens case-insensitively so the same word doesn't appear twice
    for token in tokens:
        lower_token = token.lower()
        if lower_token not in seen:
            seen.add(lower_token)
            deduped.append(token)
            
    return " ".join(deduped)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/query_expansion_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    def print_log(msg):
        logging.info(msg)
        
    print_log("================================================================================")
    print_log("[STAGE 27] Build QUERY_ENRICHMENT dict + expand_query()")
    print_log(f"Started : {timestamp}")
    print_log("Script  : src/query_expansion.py")
    print_log("================================================================================")
    
    print_log("INIT     Confirming QUERY_ENRICHMENT loaded")
    
    assertions_results = []
    
    expected_keys = {f"Q{i}" for i in range(1, 11)}
    a1_pass = set(QUERY_ENRICHMENT.keys()) == expected_keys
    assertions_results.append(("1. QUERY_ENRICHMENT contains exactly 10 keys: Q1 .. Q10", a1_pass))
    
    a2_pass = True
    for k, v in QUERY_ENRICHMENT.items():
        if not all(field in v for field in ["synonyms", "location_hint", "color_hint", "object_hint"]):
            a2_pass = False
            break
    assertions_results.append(("2. Every entry has all 4 required fields", a2_pass))
    
    print_log("START    Running self-test for all 10 queries")
    
    a3_pass = True
    a4_pass = True
    
    for q_id, raw_q in RAW_QUERIES.items():
        try:
            expanded = expand_query(q_id, raw_q)
            trunc_expanded = expanded if len(expanded) <= 120 else expanded[:117] + "..."
            print_log(f"PROG     {q_id} -> {trunc_expanded}")
            
            if not expanded:
                a3_pass = False
            if "  " in expanded or expanded != expanded.strip():
                a4_pass = False
        except Exception as e:
            a3_pass = False
            a4_pass = False
            print_log(f"FAIL     {q_id} expansion threw exception: {e}")
            
    assertions_results.append(("3. expand_query() returns a non-empty string for all 10 query IDs", a3_pass))
    assertions_results.append(("4. No query's expanded string contains double spaces or leading/trailing whitespace", a4_pass))
    
    a5_pass = False
    try:
        expand_query("Q99", "test")
    except KeyError:
        a5_pass = True
    except Exception:
        pass
    assertions_results.append(("5. expand_query(\"Q99\", \"test\") raises KeyError", a5_pass))
    
    a6_pass = all("kitchen" in QUERY_ENRICHMENT[q]["location_hint"] for q in ["Q1", "Q2", "Q8"])
    assertions_results.append(("6. location_hint for Q1, Q2, Q8 all contain \"kitchen\"", a6_pass))
    
    a7_pass = "yellow" in QUERY_ENRICHMENT["Q6"]["color_hint"]
    assertions_results.append(("7. color_hint for Q6 contains \"yellow\"", a7_pass))
    
    q10_colors = QUERY_ENRICHMENT["Q10"]["color_hint"]
    a8_pass = "red" in q10_colors or "green" in q10_colors
    assertions_results.append(("8. color_hint for Q10 contains at least one of [\"red\", \"green\"]", a8_pass))
    
    print_log("-" * 80)
    print_log("VALIDATION")
    all_pass = True
    for desc, passed in assertions_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print_log(f"  {status} | {desc}")
    
    if all_pass:
        print_log("DONE     All 8 assertions PASS")
    else:
        print_log("DONE     Validation failures detected")
        sys.exit(1)
        
    file_size = os.path.getsize(os.path.abspath(__file__))
    print_log("-" * 80)
    print_log("ARTIFACTS")
    print_log(f"  src/query_expansion.py        {file_size} B")
    print_log("================================================================================")
    print_log(f"[STAGE 27] COMPLETE — {datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
    print_log("================================================================================")


import json
import logging

def load_stream_map(faiss_row_map_path: str = "index/indexing/faiss_row_map.jsonl") -> dict[str, str]:
    """
    Reads faiss_row_map.jsonl and returns:
        { frame_id: stream_name }
    for every record in the file.
    Raises FileNotFoundError if path does not exist.
    Raises KeyError with a clear message if expected fields are missing
    in the first record.
    """
    if not os.path.exists(faiss_row_map_path):
        raise FileNotFoundError(f"File not found: {faiss_row_map_path}")
        
    stream_map = {}
    is_first = True
    
    with open(faiss_row_map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            record = json.loads(line)
            if is_first:
                if "frame_id" not in record or "stream_name" not in record:
                    raise KeyError("Expected fields frame_id and stream_name missing in the first record")
                is_first = False
                
            stream_map[record["frame_id"]] = record["stream_name"]
            
    if is_first:
        raise ValueError(f"File is empty (0 non-blank lines): {faiss_row_map_path}")
        
    return stream_map


# fixed-camera room streams that are off-topic for kitchen/appliance queries
_WRONG_ROOM_FIXED_CAMS = frozenset(["living1", "living2", "meeting", "reading"])
# queries where the action happens in the kitchen — penalise wrong-room fixed cams
_KITCHEN_FIRST_QUERIES = frozenset(["Q1", "Q2", "Q8"])


def filter_by_location(
    results: list[tuple[str, float]],
    query_id: str,
    stream_map: dict[str, str]
) -> list[tuple[str, float]]:
    """
    Down-weights scores for frames from unexpected locations.

    Step 56 update for Q1, Q2, Q8:
      - All member egocentric streams → full weight (relevant frames are egocentric)
      - Kitchen fixed-cam             → full weight (may still have signal)
      - Wrong-room fixed-cams         → 50% penalty (living1/living2/meeting/reading)

    All other queries use the original location_hint partial-match logic.

    Args:
        results    : list of (frame_id, score) — any length, pre-sorted or not
        query_id   : "Q1" .. "Q10" or "CUSTOM"
        stream_map : dict returned by load_stream_map()

    Returns:
        list of (frame_id, adjusted_score), sorted descending by score.
    """
    if query_id not in QUERY_ENRICHMENT:
        return sorted(results, key=lambda x: x[1], reverse=True)

    location_hint = QUERY_ENRICHMENT[query_id]["location_hint"]

    # queries with "any" location hint don't need filtering
    if location_hint == ["any"]:
        return sorted(results, key=lambda x: x[1], reverse=True)

    adjusted_results = []

    for frame_id, score in results:
        stream_name = stream_map.get(frame_id, None)
        if stream_name is None:
            adjusted_results.append((frame_id, score))
            continue

        if query_id in _KITCHEN_FIRST_QUERIES:
            # penalise wrong-room fixed cams but leave egocentric streams untouched
            if stream_name in _WRONG_ROOM_FIXED_CAMS:
                adjusted_results.append((frame_id, score * 0.5))
            else:
                adjusted_results.append((frame_id, score))
        else:
            # for other queries, check if the stream matches any location hint
            is_match = any(
                hint.lower() in stream_name.lower()
                or stream_name.lower() in hint.lower()
                for hint in location_hint
            )
            if is_match:
                adjusted_results.append((frame_id, score))
            else:
                # halve the score for frames from unexpected locations
                adjusted_results.append((frame_id, score * 0.5))

    return sorted(adjusted_results, key=lambda x: x[1], reverse=True)
