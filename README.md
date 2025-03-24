# WARMISO

For bm25: python bm25.py --llm llama3-70b --Q2D no --save_path logs/result_bm25.json
  
For DPR:  python DPR.py --llm llama3-70b --Q2D no --save_path logs/result_dpr.json

For Q2D+bm25: python bm25.py --llm llama3-70b --Q2D yes --save_path logs/result_q2d_bm25.json
    
For Q2D+DPR:  python DPR.py --llm llama3-70b --Q2D yes --save_path logs/result_q2d_dpr.json

For Ours:  python main_multi_hop.py --llm llama3-70b --save_path logs/result_multi_hop.json

Notes:
  1. The "--llm" can be gpt-4o, deepseek-chat, or others
  2. some test sample can be see in "data/*", please contact us for all.

