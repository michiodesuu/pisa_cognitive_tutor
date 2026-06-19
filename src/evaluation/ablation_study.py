"""
ablation_study.py
──────────────────
Analyzes the cognitive depth scores in the SQLite database grouped by 
the ablation configuration used during the chat session.
"""

import sqlite3
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

def run_ablation():
    db_path = Path("data/chat_logs/sessions.db")
    if not db_path.exists():
        print("No database found at data/chat_logs/sessions.db")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get all scores joined with session ablation config
        cursor.execute("""
            SELECT s.ablation_config, sc.dominant_icap
            FROM sessions s
            JOIN scores sc ON s.session_id = sc.session_id
        """)
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        print("No scores found. Have you run the analyzer pipeline yet?")
        return
    
    if not rows:
        print("No scored turns found in the database.")
        return

    # Group by ablation config
    results = defaultdict(lambda: {"Passive": 0, "Active": 0, "Constructive": 0, "Interactive": 0, "Total": 0})
    
    for ablation_str, icap in rows:
        try:
            cfg = json.loads(ablation_str) if ablation_str else {}
        except json.JSONDecodeError:
            cfg = {}
            
        # Classify the experimental group
        if not cfg or all(cfg.values()):
            group = "Base System (All Features Enabled)"
        elif not cfg.get("use_knowledge_graph", True):
            group = "Ablation: No Knowledge Graph"
        elif not cfg.get("use_rag", True):
            group = "Ablation: Pure LLM (No RAG)"
        elif not cfg.get("use_hybrid_search", True):
            group = "Ablation: Dense Search Only"
        else:
            group = f"Custom Ablation: {cfg}"
            
        if icap in results[group]:
            results[group][icap] += 1
        results[group]["Total"] += 1
            
    print("\n" + "="*50)
    print(" ABLATION STUDY RESULTS: COGNITIVE DEPTH (ICAP)")
    print("="*50)
    
    for group, counts in results.items():
        total = counts["Total"]
        if total == 0: continue
        
        print(f"\n[Group]: {group}")
        print(f"   Total Scored Turns: {total}")
        
        for level in ["Passive", "Active", "Constructive", "Interactive"]:
            pct = (counts[level] / total) * 100
            
            # Simple terminal bar chart
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            print(f"   {level:>12}: {counts[level]:>3} ({pct:>4.1f}%) | {bar}")
            
    print("\n" + "="*50)
    conn.close()

if __name__ == "__main__":
    run_ablation()
